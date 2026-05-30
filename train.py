"""Main training entrypoint for the CAIP project."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.data import get_dataloaders
from src.models import CAIPModel
from src.training import Trainer
from src.utils import configure_hf_downloads, count_parameters, get_default_device, plot_memory_stats, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CAIP multimodal sarcasm detector.")

    parser.add_argument("--train_file", "--train-jsonl", dest="train_file", type=str, required=True)
    parser.add_argument("--val_file", "--val-jsonl", dest="val_file", type=str, required=True)
    parser.add_argument("--test_file", "--test-jsonl", dest="test_file", type=str, required=True)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--entropy_temp", type=float, default=1.0)
    parser.add_argument("--memory_size", type=int, default=512)

    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_interactive", "no_entropy", "no_memory", "no_affective"],
    )
    parser.add_argument("--use-interactive-encoding", dest="use_interactive_encoding", type=int, choices=[0, 1])
    parser.add_argument("--use-entropy-guided", dest="use_entropy_guided", type=int, choices=[0, 1])
    parser.add_argument("--use-test-memory", dest="use_test_memory", type=int, choices=[0, 1])
    parser.add_argument("--use-affective-gap", dest="use_affective_gap", type=int, choices=[0, 1])

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--monitor_metric",
        "--monitor-metric",
        dest="monitor_metric",
        type=str,
        default="f1",
        choices=["f1", "acc", "p", "r", "ece", "easy_acc", "hard_acc"],
    )

    parser.add_argument("--clip_model_name", "--clip-model-name", dest="clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--hf_cache_dir", "--hf-cache-dir", dest="hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", "--hf-token", dest="hf_token", type=str, default=None)
    parser.add_argument("--local_files_only", "--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-2)
    parser.add_argument("--experiment_name", "--experiment-name", dest="experiment_name", type=str, default="caip")
    parser.add_argument(
        "--run_root",
        "--run-root",
        dest="run_root",
        type=str,
        default=None,
        help="Root directory for all experiment runs. Default: <project>/runs/train",
    )
    parser.add_argument(
        "--run_id",
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Run identifier under experiment_name. Default: auto timestamp.",
    )
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir", type=str, default=None)
    parser.add_argument("--history_path", "--history-path", dest="history_path", type=str, default=None)
    parser.add_argument("--analysis_cache", "--analysis-cache", dest="analysis_cache", type=str, default=None)
    parser.add_argument("--memory_plot", "--memory-plot", dest="memory_plot", type=str, default=None)
    parser.add_argument("--metrics_output", "--metrics-output", dest="metrics_output", type=str, default=None)

    return parser.parse_args()


def resolve_ablation_flags(ablation: str) -> Dict[str, bool]:
    flags = {
        "use_interactive_encoding": True,
        "use_entropy_guided": True,
        "use_test_memory": True,
        "use_affective_gap": True,
    }

    if ablation == "no_interactive":
        flags["use_interactive_encoding"] = False
    elif ablation == "no_entropy":
        flags["use_entropy_guided"] = False
    elif ablation == "no_memory":
        flags["use_test_memory"] = False
    elif ablation == "no_affective":
        flags["use_affective_gap"] = False

    return flags


def apply_flag_overrides(args: argparse.Namespace, flags: Dict[str, bool]) -> Dict[str, bool]:
    override_map = {
        "use_interactive_encoding": args.use_interactive_encoding,
        "use_entropy_guided": args.use_entropy_guided,
        "use_test_memory": args.use_test_memory,
        "use_affective_gap": args.use_affective_gap,
    }

    for key, value in override_map.items():
        if value is not None:
            flags[key] = bool(value)
    return flags


def print_parameter_summary(model: torch.nn.Module) -> None:
    stats = count_parameters(model)
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters: {stats['frozen']:,}")


def load_best_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


@torch.no_grad()
def collect_analysis_cache(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    reset_memory: bool = True,
) -> Dict[str, List[Any]]:
    model.eval()

    test_memory = getattr(model, "test_memory", None)
    if reset_memory and test_memory is not None:
        if hasattr(test_memory, "reset"):
            test_memory.reset()
        elif hasattr(test_memory, "memory"):
            test_memory.memory = {0: [], 1: []}

    cache: Dict[str, List[Any]] = {
        "entropy": [],
        "sem_gap": [],
        "aff_gap": [],
        "fusion_weights": [],
        "predictions": [],
        "labels": [],
    }

    # Keep test-time memory updates aligned with sample order.
    # 保持测试记忆按样本顺序更新，避免 batch 化评估破坏检索语义。
    sequential_memory_update = bool(
        getattr(model, "use_test_memory", False) and getattr(data_loader, "batch_size", None) == 1
    )

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            update_memory=sequential_memory_update,
        )

        # Save per-sample analysis values so analyze.py can validate EGDF and gap behavior.
        cache["entropy"].extend(outputs["entropy"].view(-1).detach().cpu().tolist())
        cache["sem_gap"].extend(outputs["sem_gap"].view(-1).detach().cpu().tolist())
        cache["aff_gap"].extend(outputs["aff_gap"].view(-1).detach().cpu().tolist())
        cache["fusion_weights"].extend(outputs["fusion_weights"].detach().cpu().tolist())
        cache["predictions"].extend(outputs["predictions"].detach().cpu().tolist())
        cache["labels"].extend(labels.detach().cpu().tolist())

    return cache


def save_pickle(obj: Any, file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)
    return path


def save_json(obj: Dict[str, Any], file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)
    return path


def resolve_run_artifacts(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else (project_root / "runs" / "train")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / args.experiment_name / run_id

    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else run_dir / "checkpoints"
    )
    history_path = Path(args.history_path).expanduser().resolve() if args.history_path else run_dir / "history.json"
    analysis_cache = (
        Path(args.analysis_cache).expanduser().resolve() if args.analysis_cache else run_dir / "analysis_cache.pkl"
    )
    memory_plot = Path(args.memory_plot).expanduser().resolve() if args.memory_plot else run_dir / "memory_stats.png"
    metrics_output = Path(args.metrics_output).expanduser().resolve() if args.metrics_output else run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "run_id": run_id,
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "history_path": history_path,
        "analysis_cache": analysis_cache,
        "memory_plot": memory_plot,
        "metrics_output": metrics_output,
        "manifest_path": manifest_path,
    }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    configure_hf_downloads(hf_token=args.hf_token, disable_xet=True)

    device = get_default_device()
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")

    artifacts = resolve_run_artifacts(args)
    print(f"Run dir: {artifacts['run_dir']}")

    ablation_flags = resolve_ablation_flags(args.ablation)
    ablation_flags = apply_flag_overrides(args, ablation_flags)
    print(f"Ablation: {args.ablation}")
    print(json.dumps(ablation_flags, indent=2))

    dataloaders = get_dataloaders(
        train_jsonl=args.train_file,
        val_jsonl=args.val_file,
        test_jsonl=args.test_file,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
    )

    model = CAIPModel(
        clip_model_name=args.clip_model_name,
        lora_rank=args.lora_rank,
        entropy_temp=args.entropy_temp,
        memory_capacity=args.memory_size,
        use_interactive_encoding=ablation_flags["use_interactive_encoding"],
        use_entropy_guided=ablation_flags["use_entropy_guided"],
        use_test_memory=ablation_flags["use_test_memory"],
        use_affective_gap=ablation_flags["use_affective_gap"],
        hf_cache_dir=args.hf_cache_dir,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
    )
    print_parameter_summary(model)

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(artifacts["checkpoint_dir"]),
        history_path=str(artifacts["history_path"]),
        monitor_metric=args.monitor_metric,
    )

    history = trainer.train()
    print(f"Training epochs completed: {len(history)}")

    last_epoch = history[-1] if history else {}
    if "train_sentiment_contrast_loss" in last_epoch:
        print("Train contrastive check:")
        print(
            json.dumps(
                {
                    "train_sentiment_contrast_loss": last_epoch.get("train_sentiment_contrast_loss"),
                    "train_sentiment_loss_head": last_epoch.get("train_sentiment_loss_head"),
                    "train_sentiment_loss_tail": last_epoch.get("train_sentiment_loss_tail"),
                    "train_sentiment_loss_trend": last_epoch.get("train_sentiment_loss_trend"),
                    "train_sentiment_loss_nonzero_rate": last_epoch.get("train_sentiment_loss_nonzero_rate"),
                    "train_hard_contrastive_loss": last_epoch.get("train_hard_contrastive_loss"),
                    "train_hard_loss_nonzero_rate": last_epoch.get("train_hard_loss_nonzero_rate"),
                },
                indent=2,
            )
        )

    best_checkpoint_path = artifacts["checkpoint_dir"] / "best_model.pt"
    load_best_checkpoint(model, best_checkpoint_path, device)

    test_metrics = trainer.evaluate(dataloaders["test"], reset_memory=True)
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    analysis_cache = collect_analysis_cache(model, dataloaders["test"], device=device, reset_memory=True)
    analysis_cache_path = save_pickle(analysis_cache, str(artifacts["analysis_cache"]))
    memory_plot_path = plot_memory_stats(model.test_memory, output_path=str(artifacts["memory_plot"]))

    metrics_payload: Dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "ablation": args.ablation,
        "lora_rank": args.lora_rank,
        "entropy_temp": args.entropy_temp,
        "memory_size": args.memory_size,
        "device": str(device),
        "run_id": str(artifacts["run_id"]),
        "run_dir": str(artifacts["run_dir"]),
        "history_path": str(artifacts["history_path"]),
        "checkpoint_dir": str(artifacts["checkpoint_dir"]),
        "test_metrics": test_metrics,
        "f1": test_metrics["f1"],
        "best_checkpoint": str(best_checkpoint_path),
        "analysis_cache": str(analysis_cache_path),
        "memory_plot": str(memory_plot_path),
    }
    metrics_path = save_json(metrics_payload, str(artifacts["metrics_output"]))

    run_manifest: Dict[str, Any] = {
        "script": "train.py",
        "experiment_name": args.experiment_name,
        "ablation": args.ablation,
        "run_id": str(artifacts["run_id"]),
        "run_dir": str(artifacts["run_dir"]),
        "artifacts": {
            "checkpoint_dir": str(artifacts["checkpoint_dir"]),
            "history_path": str(artifacts["history_path"]),
            "metrics_output": str(metrics_path),
            "analysis_cache": str(analysis_cache_path),
            "memory_plot": str(memory_plot_path),
            "best_checkpoint": str(best_checkpoint_path),
        },
        "data": {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "test_file": args.test_file,
        },
        "ablation_flags": ablation_flags,
        "hyperparameters": {
            "lora_rank": args.lora_rank,
            "entropy_temp": args.entropy_temp,
            "memory_size": args.memory_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "monitor_metric": args.monitor_metric,
            "seed": args.seed,
        },
    }
    manifest_path = save_json(run_manifest, str(artifacts["manifest_path"]))

    print(f"Saved checkpoint to: {best_checkpoint_path}")
    print(f"Saved history to: {artifacts['history_path']}")
    print(f"Saved analysis cache to: {analysis_cache_path}")
    print(f"Saved memory plot to: {memory_plot_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved run manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
