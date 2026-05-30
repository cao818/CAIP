"""Training utilities for the CAIP project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.utils.metrics import expected_calibration_error


class Trainer:
    """Trainer for CAIP model optimization and evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "checkpoints",
        history_path: Optional[str] = "logs/train_history.json",
        monitor_metric: str = "f1",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.monitor_metric = monitor_metric
        self.early_stopping_patience = early_stopping_patience

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not self.trainable_params:
            raise ValueError("No trainable parameters found. Check requires_grad flags before creating Trainer.")

        self.optimizer = AdamW(self.trainable_params, lr=learning_rate, weight_decay=weight_decay)

        total_steps = max(len(self.train_loader) * self.num_epochs, 1)
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_model.pt"

        self.history_path = Path(history_path) if history_path is not None else None
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_metric = float("-inf")
        self.no_improvement_epochs = 0

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "pixel_values": batch["pixel_values"].to(self.device),
            "labels": batch["label"].to(self.device),
        }

    def _reset_test_memory(self) -> None:
        memory = getattr(self.model, "test_memory", None)
        if memory is None:
            return
        if hasattr(memory, "reset"):
            memory.reset()
            return
        if hasattr(memory, "memory"):
            memory.memory = {0: [], 1: []}

    def _compute_difficulty_metrics(
        self,
        labels: List[int],
        predictions: List[int],
        entropies: List[float],
    ) -> Dict[str, float]:
        if not labels:
            return {"easy_acc": 0.0, "hard_acc": 0.0}

        # Use the entropy median as a simple uncertainty-based splitter.
        # 使用熵的中位数区分 easy / hard，便于观察模型在不同难度样本上的稳定性。
        entropy_tensor = torch.tensor(entropies, dtype=torch.float32)
        threshold = float(entropy_tensor.median().item())

        easy_indices = [idx for idx, entropy in enumerate(entropies) if entropy <= threshold]
        hard_indices = [idx for idx, entropy in enumerate(entropies) if entropy > threshold]

        metrics = {
            "easy_acc": self._safe_accuracy(labels, predictions, easy_indices),
            "hard_acc": self._safe_accuracy(labels, predictions, hard_indices),
        }
        return metrics

    @staticmethod
    def _safe_accuracy(labels: List[int], predictions: List[int], indices: List[int]) -> float:
        if not indices:
            return 0.0
        subset_labels = [labels[idx] for idx in indices]
        subset_predictions = [predictions[idx] for idx in indices]
        return float(accuracy_score(subset_labels, subset_predictions))

    def _save_history(self) -> None:
        if self.history_path is None:
            return
        with self.history_path.open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        torch.save(checkpoint, self.best_model_path)

    @staticmethod
    def _summarize_contrastive_stats(sentiment_losses: List[float], hard_losses: List[float]) -> Dict[str, float]:
        if not sentiment_losses and not hard_losses:
            return {
                "sentiment_contrast_loss": 0.0,
                "hard_contrastive_loss": 0.0,
                "sentiment_loss_head": 0.0,
                "sentiment_loss_tail": 0.0,
                "sentiment_loss_trend": 0.0,
                "hard_loss_nonzero_rate": 0.0,
                "sentiment_loss_nonzero_rate": 0.0,
            }

        def mean(values: List[float]) -> float:
            return float(sum(values) / max(len(values), 1))

        head_tail_k = max(int(len(sentiment_losses) * 0.2), 1) if sentiment_losses else 0
        sentiment_head = mean(sentiment_losses[:head_tail_k]) if head_tail_k > 0 else 0.0
        sentiment_tail = mean(sentiment_losses[-head_tail_k:]) if head_tail_k > 0 else 0.0

        hard_nonzero = sum(1 for v in hard_losses if v > 1e-12)
        sentiment_nonzero = sum(1 for v in sentiment_losses if v > 1e-12)

        return {
            "sentiment_contrast_loss": mean(sentiment_losses) if sentiment_losses else 0.0,
            "hard_contrastive_loss": mean(hard_losses) if hard_losses else 0.0,
            "sentiment_loss_head": sentiment_head,
            "sentiment_loss_tail": sentiment_tail,
            "sentiment_loss_trend": float(sentiment_tail - sentiment_head),
            "hard_loss_nonzero_rate": float(hard_nonzero / max(len(hard_losses), 1)) if hard_losses else 0.0,
            "sentiment_loss_nonzero_rate": float(sentiment_nonzero / max(len(sentiment_losses), 1))
            if sentiment_losses
            else 0.0,
        }

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        sentiment_losses: List[float] = []
        hard_losses: List[float] = []

        for batch in self.train_loader:
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                update_memory=False,
            )
            loss = outputs["loss"]
            loss.backward()

            # Clip only trainable parameters so the frozen CLIP backbone stays untouched.
            # 只裁剪可训练参数，避免对冻结主干做无意义的梯度处理。
            clip_grad_norm_(self.trainable_params, max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            running_loss += float(loss.item())
            num_batches += 1

            if "sentiment_contrast_loss" in outputs:
                sentiment_losses.append(float(outputs["sentiment_contrast_loss"].detach().item()))
            if "hard_contrastive_loss" in outputs:
                hard_losses.append(float(outputs["hard_contrastive_loss"].detach().item()))

        self.last_train_stats = self._summarize_contrastive_stats(sentiment_losses, hard_losses)
        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, reset_memory: bool = False) -> Dict[str, float]:
        self.model.eval()

        if reset_memory:
            self._reset_test_memory()

        all_labels: List[int] = []
        all_predictions: List[int] = []
        all_entropies: List[float] = []
        all_probabilities: List[List[float]] = []

        # Test-time memory only makes sense under sequential evaluation.
        # 测试记忆依赖顺序写入，因此仅在 batch_size=1 的评估阶段启用在线更新。
        sequential_memory_update = bool(
            getattr(self.model, "use_test_memory", False) and getattr(data_loader, "batch_size", None) == 1
        )

        for batch in data_loader:
            batch = self._move_batch_to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                update_memory=sequential_memory_update,
            )

            labels = batch["labels"].detach().cpu().tolist()
            predictions = outputs["predictions"].detach().cpu().tolist()
            entropies = outputs["entropy"].detach().view(-1).cpu().tolist()
            probabilities = outputs["probabilities"].detach().cpu().tolist()

            all_labels.extend(labels)
            all_predictions.extend(predictions)
            all_entropies.extend(entropies)
            all_probabilities.extend(probabilities)

        if not all_labels:
            return {"acc": 0.0, "f1": 0.0, "p": 0.0, "r": 0.0, "ece": 0.0, "easy_acc": 0.0, "hard_acc": 0.0}

        metrics = {
            "acc": float(accuracy_score(all_labels, all_predictions)),
            "f1": float(f1_score(all_labels, all_predictions, zero_division=0)),
            "p": float(precision_score(all_labels, all_predictions, zero_division=0)),
            "r": float(recall_score(all_labels, all_predictions, zero_division=0)),
            "ece": expected_calibration_error(
                probabilities=torch.tensor(all_probabilities, dtype=torch.float32),
                labels=torch.tensor(all_labels, dtype=torch.long),
            ),
        }
        metrics.update(self._compute_difficulty_metrics(all_labels, all_predictions, all_entropies))
        return metrics

    def train(self) -> List[Dict[str, Any]]:
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch()
            epoch_record: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            should_stop = False
            if hasattr(self, "last_train_stats"):
                epoch_record.update({f"train_{key}": value for key, value in self.last_train_stats.items()})

            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, reset_memory=True)
                epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})

                current_metric = float(val_metrics.get(self.monitor_metric, val_metrics["f1"]))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.no_improvement_epochs = 0
                    self._save_checkpoint(epoch, val_metrics)
                    epoch_record["best_model_saved"] = True
                else:
                    self.no_improvement_epochs += 1
                    epoch_record["best_model_saved"] = False
                    epoch_record["no_improvement_epochs"] = self.no_improvement_epochs

                # Stop training when validation performance has not improved
                # for several consecutive epochs.
                # 当验证指标连续多个 epoch 没有提升时，提前停止训练。
                if self.no_improvement_epochs >= self.early_stopping_patience:
                    should_stop = True
                    epoch_record["early_stopped"] = True
            else:
                if epoch == self.num_epochs:
                    self.best_metric = -train_loss
                    self._save_checkpoint(epoch, {"train_loss": train_loss})
                epoch_record["best_model_saved"] = epoch == self.num_epochs

            self.history.append(epoch_record)
            self._save_history()

            if should_stop:
                break

        return self.history
