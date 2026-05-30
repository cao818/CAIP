"""Runtime utilities for reproducibility and device management."""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """设置随机种子，提升实验可复现性。

    覆盖范围：
    - Python 内置 random
    - NumPy 随机数
    - PyTorch CPU 随机数
    - PyTorch CUDA 随机数（若可用）
    并尽可能开启确定性后端设置（可能会牺牲少量性能）。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic behavior when possible for better reproducibility.
    # 在可行范围内启用确定性后端，减少多次运行之间的随机波动。
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_default_device() -> torch.device:
    """自动选择默认运行设备：优先 CUDA，否则使用 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数量（总参数 / 可训练参数 / 冻结参数）。"""

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }


def configure_hf_downloads(hf_token: Optional[str] = None, disable_xet: bool = True) -> None:
    """Configures Hugging Face download behavior for restricted environments.

    The current training environment may fail on the `xet`/CAS download path.
    We disable it by default and optionally inject a user token for gated or
    rate-limited downloads.
    """

    # Hugging Face Hub 在部分环境下会走 xet/CAS 的下载路径，这类路径可能
    # 依赖额外的网络能力或系统特性，导致下载失败，因此默认关闭。
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    # 如果传入 token，则注入到常见环境变量中，让 transformers/huggingface_hub
    # 在访问 gated model、或触发 rate-limit 时能正常鉴权。
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
