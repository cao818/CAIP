"""Visualization script for CAIP analysis cache."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_cache(cache_path: str) -> Any:
    """从磁盘加载由训练/评估阶段保存的 analysis_cache.pkl。

    cache 的结构在不同版本里可能略有差异（dict / list[dict] 都可能出现），
    这里保持最大兼容性，后续通过 extract_field 做字段提取与降级。
    """
    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Analysis cache not found: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def _to_numpy_array(value: Any) -> np.ndarray:
    """把输入值尽可能转换为 float32 的 numpy 数组。

    支持：
    - torch.Tensor（detach/cpu/numpy）
    - numpy / list / 标量等可被 np.asarray 接受的对象
    """
    if value is None:
        return np.array([], dtype=np.float32)

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float32)

    if array.size == 0:
        return np.array([], dtype=np.float32)
    return array


def _flatten_numeric(value: Any) -> np.ndarray:
    """把数值字段拍平成一维向量，便于做统计与可视化。"""
    array = _to_numpy_array(value)
    if array.size == 0:
        return array
    return array.reshape(-1)


def _extract_from_dict(cache: dict, candidate_keys: Sequence[str]) -> Any:
    """从 dict cache 中按候选 key 列表依次尝试提取字段。"""
    for key in candidate_keys:
        if key in cache:
            return cache[key]
    return None


def _extract_from_sequence(cache: Sequence[Any], candidate_keys: Sequence[str]) -> Any:
    """从 list/tuple[dict] 结构中提取字段并汇总成一个列表。"""
    values: List[Any] = []
    for item in cache:
        if not isinstance(item, dict):
            continue
        for key in candidate_keys:
            if key in item:
                values.append(item[key])
                break
    if not values:
        return None
    return values


def extract_field(cache: Any, candidate_keys: Sequence[str]) -> Any:
    """统一字段提取入口，兼容 dict 与 list[dict] 两种 cache 结构。"""
    if isinstance(cache, dict):
        return _extract_from_dict(cache, candidate_keys)
    if isinstance(cache, (list, tuple)):
        return _extract_from_sequence(cache, candidate_keys)
    return None


def extract_entropy(cache: Any) -> np.ndarray:
    """提取样本熵（如果缺失则返回空数组）。"""
    entropy = extract_field(cache, ("entropy", "entropies", "sample_entropy"))
    return _flatten_numeric(entropy)


def extract_alpha(cache: Any, entropy: np.ndarray) -> np.ndarray:
    """提取 alpha（融合强度/权重）并提供降级策略。

    优先级：
    1) cache 中显式提供 alpha
    2) 若提供 fusion_weights，则用其最大值近似 alpha
    3) 若都没有，但 entropy 可用，则用 1 - entropy 作为近似
    """
    alpha = extract_field(cache, ("alpha", "alphas", "fusion_alpha", "adaptive_alpha"))
    alpha_array = _to_numpy_array(alpha)

    if alpha_array.size == 0:
        fusion_weights = extract_field(cache, ("fusion_weights", "weights", "alphas_per_branch"))
        weight_array = _to_numpy_array(fusion_weights)
        if weight_array.ndim >= 2:
            alpha_array = weight_array.max(axis=-1)
        else:
            alpha_array = _flatten_numeric(weight_array)

    if alpha_array.size == 0 and entropy.size > 0:
        alpha_array = 1.0 - entropy

    return _flatten_numeric(alpha_array)


def extract_gap(cache: Any, candidate_keys: Sequence[str]) -> np.ndarray:
    """提取 gap（语义/情感不一致性强度）字段并拍平。"""
    return _flatten_numeric(extract_field(cache, candidate_keys))


def align_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    """将多个一维数组对齐到同一长度（截断到最短的有效长度）。

    目的：避免不同字段长度不一致导致绘图报错（常见于部分字段缺失或拼接方式不同）。
    """
    valid_lengths = [len(array) for array in arrays if array.size > 0]
    if not valid_lengths:
        return [np.array([], dtype=np.float32) for _ in arrays]
    min_len = min(valid_lengths)
    return [array[:min_len] if array.size > 0 else np.array([], dtype=np.float32) for array in arrays]


def plot_entropy_distribution(ax: plt.Axes, entropy: np.ndarray) -> None:
    """绘制熵的直方图 + KDE，用于观察模型不确定性分布。"""
    sns.histplot(entropy, bins=30, kde=True, color="#4C72B0", ax=ax)
    ax.set_title("Entropy Distribution")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")


def plot_alpha_vs_entropy(ax: plt.Axes, alpha: np.ndarray, entropy: np.ndarray) -> None:
    """绘制 alpha 与 entropy 的回归关系，用于验证融合权重是否随不确定性变化。"""
    sns.regplot(
        x=entropy,
        y=alpha,
        scatter_kws={"alpha": 0.6, "s": 30},
        line_kws={"color": "#C44E52", "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Alpha vs Entropy")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Alpha")


def plot_gap_distribution(ax: plt.Axes, sem_gap: np.ndarray, aff_gap: np.ndarray) -> None:
    """绘制语义 gap 与情感 gap 的分布，便于观察两类不一致性强度范围。"""
    sns.histplot(sem_gap, bins=30, kde=True, color="#55A868", label="Semantic Gap", ax=ax, stat="density", alpha=0.5)
    sns.histplot(aff_gap, bins=30, kde=True, color="#8172B2", label="Affective Gap", ax=ax, stat="density", alpha=0.5)
    ax.set_title("Gap Distribution")
    ax.set_xlabel("Gap Value")
    ax.set_ylabel("Density")
    ax.legend()


def plot_gap_correlation(ax: plt.Axes, sem_gap: np.ndarray, aff_gap: np.ndarray) -> None:
    """绘制 sem_gap 与 aff_gap 的相关性散点/回归，并标注皮尔逊相关系数。"""
    sns.regplot(
        x=sem_gap,
        y=aff_gap,
        scatter_kws={"alpha": 0.6, "s": 30},
        line_kws={"color": "#DD8452", "linewidth": 2},
        ax=ax,
    )
    corr = float(np.corrcoef(sem_gap, aff_gap)[0, 1]) if len(sem_gap) > 1 else 0.0
    ax.set_title("Gap Correlation")
    ax.set_xlabel("Semantic Gap")
    ax.set_ylabel("Affective Gap")
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, va="top")


def validate_nonempty(name: str, values: np.ndarray) -> None:
    """字段校验：如果关键字段无法抽取，直接抛错提示 cache 格式不兼容。"""
    if values.size == 0:
        raise ValueError(f"Could not extract '{name}' from analysis cache.")


def create_analysis_figure(cache: Any, output_path: str) -> Path:
    """从 cache 抽取字段并生成四宫格分析图。

    图像内容：
    1) 熵分布
    2) alpha vs 熵（融合权重与不确定性的关系）
    3) 语义/情感 gap 分布
    4) 语义/情感 gap 相关性
    """
    entropy = extract_entropy(cache)
    sem_gap = extract_gap(cache, ("sem_gap", "semantic_gap", "semantic_gaps"))
    aff_gap = extract_gap(cache, ("aff_gap", "affective_gap", "affective_gaps"))

    validate_nonempty("entropy", entropy)
    validate_nonempty("sem_gap", sem_gap)
    validate_nonempty("aff_gap", aff_gap)

    alpha = extract_alpha(cache, entropy)
    validate_nonempty("alpha", alpha)

    # 对齐长度，避免不同字段长度不一致导致绘图函数报错。
    entropy, alpha = align_arrays(entropy, alpha)
    sem_gap, aff_gap = align_arrays(sem_gap, aff_gap)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_entropy_distribution(axes[0], entropy)
    plot_alpha_vs_entropy(axes[1], alpha, entropy)
    plot_gap_distribution(axes[2], sem_gap, aff_gap)
    plot_gap_correlation(axes[3], sem_gap, aff_gap)

    fig.suptitle("CAIP Analysis", fontsize=18)
    fig.tight_layout()

    # 统一使用绝对路径保存，确保脚本从任意工作目录运行都能落盘到预期位置。
    save_path = Path(output_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Visualize CAIP analysis cache.")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="analysis_cache.pkl",
        help="Path to the saved analysis_cache.pkl file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="caip_analysis.png",
        help="Path to save the generated figure.",
    )
    return parser


def main() -> None:
    """脚本入口：读取 cache，生成图片并输出保存路径。"""
    parser = build_argparser()
    args = parser.parse_args()
    cache = load_cache(args.cache_path)
    output_path = create_analysis_figure(cache, args.output_path)
    print(f"Saved analysis figure to: {output_path}")


if __name__ == "__main__":
    main()
