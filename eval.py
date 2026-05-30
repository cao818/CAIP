"""Standalone evaluation entrypoint for the CAIP project."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.data import get_dataloaders
from src.models import CAIPModel
from src.training import Trainer
from src.utils import configure_hf_downloads, get_default_device, plot_memory_stats, set_random_seed
from train import (
    apply_flag_overrides,
    collect_analysis_cache,
    load_best_checkpoint,
    resolve_ablation_flags,
    save_json,
    save_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CAIP checkpoint.")

    parser.add_argument("--test_file", "--test-jsonl", dest="test_file", type=str, required=True)
    parser.add_argument("--val_file", "--val-jsonl", dest="val_file", type=str, default=None)

    parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--clip_model_name", "--clip-model-name", dest="clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--hf_cache_dir", "--hf-cache-dir", dest="hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", "--hf-token", dest="hf_token", type=str, default=None)
    parser.add_argument("--local_files_only", "--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

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

    parser.add_argument("--experiment_name", "--experiment-name", dest="experiment_name", type=str, default="caip_eval")
    parser.add_argument(
        "--run_root",
        "--run-root",
        dest="run_root",
        type=str,
        default=None,
        help="Root directory for evaluation runs. Default: <project>/runs/eval",
    )
    parser.add_argument(
        "--run_id",
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Run identifier under experiment_name. Default: auto timestamp.",
    )
    parser.add_argument("--metrics_output", "--metrics-output", dest="metrics_output", type=str, default=None)
    parser.add_argument("--analysis_cache", "--analysis-cache", dest="analysis_cache", type=str, default=None)
    parser.add_argument("--memory_plot", "--memory-plot", dest="memory_plot", type=str, default=None)
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> CAIPModel:
    ablation_flags = resolve_ablation_flags(args.ablation)
    ablation_flags = apply_flag_overrides(args, ablation_flags)

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
    model.to(device)
    return model


def build_trainer(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: str,
) -> Trainer:
    # Reuse the Trainer evaluation logic by providing the test loader as the
    # required loader handle. 这里不执行训练，只复用统一的 evaluate() 接口。
    return Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=None,
        test_loader=test_loader,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,
        checkpoint_dir=checkpoint_dir,
        history_path=None,
    )


def resolve_eval_artifacts(args: argparse.Namespace) -> Dict[str, Path | str]:
    project_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else (project_root / "runs" / "eval")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / args.experiment_name / run_id

    metrics_output = Path(args.metrics_output).expanduser().resolve() if args.metrics_output else run_dir / "metrics.json"
    analysis_cache = Path(args.analysis_cache).expanduser().resolve() if args.analysis_cache else run_dir / "analysis_cache.pkl"
    memory_plot = Path(args.memory_plot).expanduser().resolve() if args.memory_plot else run_dir / "memory_stats.png"
    manifest_path = run_dir / "run_manifest.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "run_id": run_id,
        "run_dir": run_dir,
        "metrics_output": metrics_output,
        "analysis_cache": analysis_cache,
        "memory_plot": memory_plot,
        "manifest_path": manifest_path,
    }


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    configure_hf_downloads(hf_token=args.hf_token, disable_xet=True)
    device = get_default_device()
    artifacts = resolve_eval_artifacts(args)
    print(f"Eval run dir: {artifacts['run_dir']}")

    dataloaders = get_dataloaders(
        train_jsonl=None,
        val_jsonl=args.val_file,
        test_jsonl=args.test_file,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
    )

    model = build_model(args, device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    load_best_checkpoint(model, checkpoint_path, device)

    trainer = build_trainer(
        model=model,
        test_loader=dataloaders["test"],
        device=device,
        checkpoint_dir=str(artifacts["run_dir"] / "_tmp_checkpoints"),
    )
    test_metrics = trainer.evaluate(dataloaders["test"], reset_memory=True)

    analysis_cache = collect_analysis_cache(model, dataloaders["test"], device=device, reset_memory=True)
    analysis_cache_path = save_pickle(analysis_cache, str(artifacts["analysis_cache"]))
    memory_plot_path = plot_memory_stats(model.test_memory, output_path=str(artifacts["memory_plot"]))

    payload: Dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "run_id": str(artifacts["run_id"]),
        "run_dir": str(artifacts["run_dir"]),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "test_metrics": test_metrics,
        "f1": test_metrics["f1"],
        "analysis_cache": str(analysis_cache_path),
        "memory_plot": str(memory_plot_path),
    }
    metrics_path = save_json(payload, str(artifacts["metrics_output"]))

    run_manifest: Dict[str, Any] = {
        "script": "eval.py",
        "experiment_name": args.experiment_name,
        "run_id": str(artifacts["run_id"]),
        "run_dir": str(artifacts["run_dir"]),
        "checkpoint": str(checkpoint_path),
        "artifacts": {
            "metrics_output": str(metrics_path),
            "analysis_cache": str(analysis_cache_path),
            "memory_plot": str(memory_plot_path),
        },
        "data": {
            "test_file": args.test_file,
            "val_file": args.val_file,
        },
        "hyperparameters": {
            "lora_rank": args.lora_rank,
            "entropy_temp": args.entropy_temp,
            "memory_size": args.memory_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "ablation": args.ablation,
    }
    manifest_path = save_json(run_manifest, str(artifacts["manifest_path"]))

    print(json.dumps(payload, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved run manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
