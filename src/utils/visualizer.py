"""Visualization helpers for CAIP utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _extract_memory_dict(memory_source: Any) -> Dict[int, List[Any]]:
    if hasattr(memory_source, "memory"):
        memory_dict = getattr(memory_source, "memory")
    elif isinstance(memory_source, Mapping):
        memory_dict = memory_source
    else:
        raise TypeError("`memory_source` must be a TestTimeMemory-like object or a mapping.")

    return {
        0: list(memory_dict.get(0, [])),
        1: list(memory_dict.get(1, [])),
    }


def _stack_channel_features(channel_features: List[Any]) -> np.ndarray:
    if not channel_features:
        return np.zeros((0, 3), dtype=np.float32)

    converted = []
    for feature in channel_features:
        if isinstance(feature, (tuple, list)) and len(feature) >= 1:
            feature = feature[0]
        if hasattr(feature, "detach"):
            feature = feature.detach().cpu().numpy()
        converted.append(np.asarray(feature, dtype=np.float32).reshape(3))
    return np.stack(converted, axis=0)


def plot_memory_stats(
    memory_source: Any,
    output_path: str = "memory_stats.png",
    dpi: int = 300,
) -> Path:
    """Visualizes memory-bank size and feature distributions.

    The memory bank stores three values per sample:
    ``[semantic_gap, affective_gap, entropy]``.
    """

    memory_dict = _extract_memory_dict(memory_source)
    nonsarcasm = _stack_channel_features(memory_dict[0])
    sarcasm = _stack_channel_features(memory_dict[1])

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    counts = [len(memory_dict[0]), len(memory_dict[1])]
    axes[0].bar(["non-sarcastic", "sarcastic"], counts, color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Memory Size by Channel")
    axes[0].set_ylabel("Sample Count")

    # Show channel-wise semantic gap distributions to inspect memory coverage.
    # 展示双通道的语义 gap 分布，便于观察记忆库覆盖是否均衡。
    if len(nonsarcasm) > 0:
        sns.kdeplot(nonsarcasm[:, 0], ax=axes[1], label="Non-sarcastic", color="#4C72B0", fill=True, alpha=0.3)
    if len(sarcasm) > 0:
        sns.kdeplot(sarcasm[:, 0], ax=axes[1], label="Sarcastic", color="#C44E52", fill=True, alpha=0.3)
    axes[1].set_title("Semantic Gap Distribution")
    axes[1].set_xlabel("Semantic Gap")
    axes[1].legend()

    if len(nonsarcasm) > 0:
        sns.kdeplot(nonsarcasm[:, 1], ax=axes[2], label="Non-sarcastic", color="#55A868", fill=True, alpha=0.3)
    if len(sarcasm) > 0:
        sns.kdeplot(sarcasm[:, 1], ax=axes[2], label="Sarcastic", color="#8172B2", fill=True, alpha=0.3)
    axes[2].set_title("Affective Gap Distribution")
    axes[2].set_xlabel("Affective Gap")
    axes[2].legend()

    if len(nonsarcasm) > 0:
        axes[3].scatter(nonsarcasm[:, 0], nonsarcasm[:, 2], alpha=0.6, label="Non-sarcastic", color="#4C72B0")
    if len(sarcasm) > 0:
        axes[3].scatter(sarcasm[:, 0], sarcasm[:, 2], alpha=0.6, label="Sarcastic", color="#C44E52")
    axes[3].set_title("Semantic Gap vs Entropy")
    axes[3].set_xlabel("Semantic Gap")
    axes[3].set_ylabel("Entropy")
    axes[3].legend()

    fig.suptitle("CAIP Memory Bank Statistics", fontsize=18)
    fig.tight_layout()

    save_path = Path(output_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
