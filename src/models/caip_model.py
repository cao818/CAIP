"""CAIP 核心模型实现。

本文件包含三个主要部分：
1. `LoRAInjector`：轻量级低秩交互注入器；
2. `AdvancedTestTimeMemory`：测试时双通道记忆库；
3. `CAIPModel`：完整的图文讽刺检测模型。

整体数据流如下：
图像/文本输入
-> 冻结 CLIP 提取基础表示
-> LoRA 交互编码
-> 语义不一致性分支 + 情感不一致性分支 + 全局分支
-> EGDF 熵引导融合
-> 测试时记忆增强（可选）
-> 分类结果与训练损失
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class LoRAInjector(nn.Module):
    """轻量级 LoRA 残差注入器。

    输入输出维度保持一致，中间通过低秩分解生成增量更新，再由门控决定
    这部分更新应以多大比例注入主表示。
    """

    def __init__(self, dim: int = 768, rank: int = 8, num_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.down_projs = nn.ModuleList([nn.Linear(dim, rank, bias=False) for _ in range(num_layers)])
        self.up_projs = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scaling = 1.0 / max(rank, 1)

        for up_proj in self.up_projs:
            nn.init.zeros_(up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for norm, down_proj, up_proj, gate in zip(self.norms, self.down_projs, self.up_projs, self.gates):
            normalized = norm(hidden)
            # 低秩分支负责学习任务相关的增量表示，门控分支决定这部分更新
            # 应该以多大比例注入当前特征。
            update = up_proj(self.dropout(F.gelu(down_proj(normalized)))) * self.scaling
            gate_value = torch.sigmoid(gate(normalized))
            hidden = hidden + gate_value * update
        return hidden


@dataclass
class _GaussianStats:
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    count: int = 0


class AdvancedTestTimeMemory:
    """改进版测试时记忆模块（融合熵排序替换 + 可选分布建模）。

    存储单位：三维特征 `[sem_gap, aff_gap, entropy]`，并额外记录该样本熵。
    - 通道 0：非讽刺
    - 通道 1：讽刺

    主要改进：
    1) 熵排序替换：在容量固定时，优先保留低熵样本；高熵样本会被动态驱逐。
    2) 熵感知 KNN：检索时同时考虑距离权重与邻居自身熵权重。
    3) 可选高斯先验：在线估计每类三维签名分布，查询时与 KNN 结果混合。
    """

    def __init__(
        self,
        capacity_per_channel: int = 512,
        use_entropy_ranking: bool = True,
        use_distribution_prior: bool = False,
        default_k: int = 5,
        entropy_temperature: float = 0.1,
        knn_metric: str = "l2",
        distribution_mix_weight: float = 0.3,
    ) -> None:
        self.capacity_per_channel = capacity_per_channel
        self.use_entropy_ranking = use_entropy_ranking
        self.use_distribution_prior = use_distribution_prior
        self.default_k = default_k
        self.entropy_temp = max(float(entropy_temperature), 1e-6)
        self.knn_metric = knn_metric
        self.distribution_mix_weight = float(distribution_mix_weight)

        self.memory: Dict[int, List[Tuple[torch.Tensor, float]]] = {0: [], 1: []}
        self.distribution_stats: Dict[int, _GaussianStats] = {0: _GaussianStats(), 1: _GaussianStats()}

    def reset(self) -> None:
        self.memory = {0: [], 1: []}
        self.distribution_stats = {0: _GaussianStats(), 1: _GaussianStats()}

    def update(self, memory_features: torch.Tensor, labels: torch.Tensor, entropies: torch.Tensor) -> int:
        if memory_features.numel() == 0:
            return 0

        features_cpu = memory_features.detach().float().cpu()
        labels_cpu = labels.detach().long().cpu()
        entropies_cpu = entropies.detach().float().cpu().view(-1)

        kept = 0
        for feature, label, entropy in zip(features_cpu, labels_cpu, entropies_cpu):
            channel = int(label.item())
            if channel not in self.memory:
                continue

            feature_3d = feature.view(3)
            entropy_value = float(entropy.item())
            channel_mem = self.memory[channel]

            if len(channel_mem) < self.capacity_per_channel:
                self._insert_sorted(channel_mem, (feature_3d, entropy_value))
                kept += 1
            else:
                if self.use_entropy_ranking:
                    worst_entropy = channel_mem[-1][1]
                    if entropy_value < worst_entropy:
                        channel_mem.pop()
                        self._insert_sorted(channel_mem, (feature_3d, entropy_value))
                        kept += 1
                else:
                    channel_mem.pop(0)
                    channel_mem.append((feature_3d, entropy_value))
                    kept += 1

            if self.use_distribution_prior:
                self._update_distribution(channel, feature_3d)

        return kept

    def query(self, memory_features: torch.Tensor, channel: Optional[int] = None, k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        device = memory_features.device
        batch_size = memory_features.size(0)
        k = k or self.default_k

        result: Dict[str, torch.Tensor] = {
            "scores": torch.zeros(batch_size, 2, device=device),
            "distances": torch.full((batch_size, 2), float("inf"), device=device),
            "counts": torch.zeros(batch_size, 2, device=device),
        }
        if self.use_distribution_prior:
            result["distribution_likelihood"] = torch.zeros(batch_size, 2, device=device)

        channels = [channel] if channel in (0, 1) else [0, 1]
        query_features = memory_features.detach().float()

        for class_idx in channels:
            bank_list = self.memory.get(class_idx, [])
            if not bank_list:
                continue

            bank_features = torch.stack([item[0] for item in bank_list], dim=0).to(device=device, dtype=query_features.dtype)
            bank_entropies = torch.tensor([item[1] for item in bank_list], device=device, dtype=query_features.dtype)
            current_k = min(k, bank_features.size(0))

            distances = self._pairwise_distance(query_features, bank_features)
            knn_dist, knn_idx = torch.topk(distances, k=current_k, largest=False, dim=-1)

            neighbor_entropies = bank_entropies[knn_idx]
            entropy_weights = torch.exp(-neighbor_entropies / self.entropy_temp)
            dist_weights = 1.0 / (knn_dist + 1e-6)
            combined = dist_weights * entropy_weights
            combined = combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            scores = (combined * dist_weights).sum(dim=-1)
            result["scores"][:, class_idx] = scores
            result["distances"][:, class_idx] = knn_dist.mean(dim=-1)
            result["counts"][:, class_idx] = float(current_k)

            if self.use_distribution_prior:
                likelihood = self._distribution_likelihood(class_idx, query_features, device=device, dtype=query_features.dtype)
                result["distribution_likelihood"][:, class_idx] = likelihood

        score_sum = result["scores"].sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_scores = result["scores"] / score_sum

        if self.use_distribution_prior:
            likelihood = result["distribution_likelihood"]
            likelihood = likelihood / likelihood.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mix = self.distribution_mix_weight
            normalized_scores = (1.0 - mix) * normalized_scores + mix * likelihood

        result["normalized_scores"] = normalized_scores
        return result

    def get_statistics(self) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        for ch in (0, 1):
            mem = self.memory.get(ch, [])
            if not mem:
                continue
            entropies = [item[1] for item in mem]
            stats[f"channel_{ch}"] = {
                "size": len(mem),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies)),
                "entropy_min": float(np.min(entropies)),
                "entropy_max": float(np.max(entropies)),
            }

            if self.use_distribution_prior:
                dist = self.distribution_stats[ch]
                if dist.count > 0 and dist.mu is not None and dist.sigma is not None:
                    stats[f"channel_{ch}"]["gaussian_mu"] = dist.mu.tolist()
                    stats[f"channel_{ch}"]["gaussian_sigma"] = dist.sigma.tolist()
        return stats

    def _insert_sorted(self, channel_mem: List[Tuple[torch.Tensor, float]], item: Tuple[torch.Tensor, float]) -> None:
        _, entropy = item
        left, right = 0, len(channel_mem)
        while left < right:
            mid = (left + right) // 2
            if channel_mem[mid][1] < entropy:
                left = mid + 1
            else:
                right = mid
        channel_mem.insert(left, item)

    def _update_distribution(self, label: int, feature: torch.Tensor) -> None:
        stats = self.distribution_stats[label]
        x = feature.detach().cpu().numpy().astype(np.float32)

        if stats.count == 0:
            stats.mu = x
            stats.sigma = np.ones_like(x, dtype=np.float32) * 0.1
            stats.count = 1
            return

        n = stats.count
        mu_old = stats.mu if stats.mu is not None else x
        mu_new = (mu_old * n + x) / (n + 1)

        sigma_old = stats.sigma if stats.sigma is not None else np.ones_like(x, dtype=np.float32) * 0.1
        delta = x - mu_old
        delta2 = x - mu_new
        sigma_new = np.sqrt((sigma_old**2 * n + delta * delta2) / (n + 1))

        stats.mu = mu_new
        stats.sigma = sigma_new
        stats.count = n + 1

    def _distribution_likelihood(self, label: int, query_features: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dist = self.distribution_stats[label]
        if dist.count <= 10 or dist.mu is None or dist.sigma is None:
            return torch.zeros(query_features.size(0), device=device, dtype=dtype)

        mu = torch.tensor(dist.mu, device=device, dtype=dtype)
        sigma = torch.tensor(dist.sigma, device=device, dtype=dtype).clamp_min(1e-6)
        log_likelihood = -0.5 * torch.sum(((query_features - mu) / sigma) ** 2 + torch.log(sigma), dim=-1)
        return log_likelihood.exp()

    def _pairwise_distance(self, query_features: torch.Tensor, bank_features: torch.Tensor) -> torch.Tensor:
        if self.knn_metric == "cosine":
            q = F.normalize(query_features, dim=-1)
            b = F.normalize(bank_features, dim=-1)
            sim = torch.matmul(q, b.t())
            return 1.0 - sim
        return torch.cdist(query_features, bank_features, p=2)


class CAIPModel(nn.Module):
    """基于 CLIP 的 CAIP 主模型。

    设计目标：
    - 使用冻结 CLIP 作为稳定 backbone；
    - 通过 LoRA 做轻量适配；
    - 显式建模语义与情感不一致性；
    - 用熵引导的方式融合多个分支；
    - 在测试阶段利用记忆库增强决策。
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 768,
        num_classes: int = 2,
        use_interactive_encoding: bool = True,
        use_entropy_guided: bool = True,
        use_test_memory: bool = True,
        use_affective_gap: bool = True,
        lora_rank: int = 8,
        entropy_temp: float = 1.0,
        memory_capacity: int = 512,
        hf_cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        local_files_only: bool = False,
        use_distribution_prior: bool = False,
        use_entropy_ranking: bool = True,
        memory_entropy_temperature: float = 0.1,
    ) -> None:
        super().__init__()

        # ============================================================
        # 1. 基础配置
        # ============================================================
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_interactive_encoding = use_interactive_encoding
        self.use_entropy_guided = use_entropy_guided
        self.use_test_memory = use_test_memory
        self.use_affective_gap = use_affective_gap
        self.entropy_temp = max(entropy_temp, 1e-6)

        # ============================================================
        # 2. 冻结 CLIP backbone
        # ============================================================
        try:
            self.clip = CLIPModel.from_pretrained(
                clip_model_name,
                cache_dir=hf_cache_dir,
                token=hf_token,
                local_files_only=local_files_only,
            )
        except OSError as error:
            raise RuntimeError(
                "Failed to load CLIPModel weights. Please set `HF_TOKEN`, disable restrictive "
                "download paths, or point `--clip_model_name` to a local CLIP directory that "
                "contains the model weights."
            ) from error

        # 冻结主干，训练时只更新 LoRA、投影层和分类头等外接模块。
        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        vision_hidden = self.clip.vision_model.config.hidden_size
        text_hidden = self.clip.text_model.config.hidden_size
        projection_dim = self.clip.config.projection_dim

        # ============================================================
        # 3. 特征投影层
        #    CLIP 输出维度与后续任务维度可能不完全一致，因此统一映射到
        #    hidden_dim，便于后续各分支共享处理。
        # ============================================================
        self.image_feature_proj = nn.Linear(projection_dim, hidden_dim)
        self.text_feature_proj = nn.Linear(projection_dim, hidden_dim)
        self.image_token_proj = nn.Linear(vision_hidden, hidden_dim)
        self.text_token_proj = nn.Linear(text_hidden, hidden_dim)

        # ============================================================
        # 4. LoRA 交互编码模块
        # ============================================================
        self.image_lora = LoRAInjector(dim=hidden_dim, rank=lora_rank, num_layers=4)
        self.text_lora = LoRAInjector(dim=hidden_dim, rank=lora_rank, num_layers=4)

        # 语义分支主要建模全局图文对齐后的不一致性，情感分支则依赖 token
        # 级跨模态注意力来刻画情绪表达上的落差。
        # ============================================================
        # 5. 不一致性建模模块
        # ============================================================
        self.correlation_conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.image_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.text_to_image_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)

        self.sentiment_fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.sentiment_fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.sentiment_relu = nn.ReLU()

        # 情感分支采用“上下文感知”的表征：将全局特征与跨模态注意力后的上下文
        # 拼接后编码成 affective embedding，再用 embedding 差异构造 aff_gap。
        self.image_affective_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.text_affective_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.image_affective_scorer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.text_affective_scorer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.affective_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ============================================================
        # 6. 三个分类头
        #    - semantic_head：语义不一致性分支
        #    - affective_head：情感不一致性分支
        #    - global_head：全局图文交互分支
        # ============================================================
        self.semantic_head = nn.Linear(hidden_dim, num_classes)
        self.affective_head = nn.Linear(hidden_dim, num_classes)
        self.global_head = nn.Linear(hidden_dim, num_classes)
        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.memory_logit_scale = nn.Parameter(torch.tensor(1.0))

        # ============================================================
        # 7. 测试时记忆库
        # ============================================================
        self.test_memory = AdvancedTestTimeMemory(
            capacity_per_channel=memory_capacity,
            use_entropy_ranking=use_entropy_ranking,
            use_distribution_prior=use_distribution_prior,
            default_k=5,
            entropy_temperature=memory_entropy_temperature,
        )

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """提取图像和文本基础特征，并在需要时执行 LoRA 交互编码。"""
        # CLIP 主干被冻结，因此这里只把它当作固定特征提取器使用。
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
            text_outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            image_embed = self.clip.visual_projection(vision_outputs.pooler_output)
            text_embed = self.clip.text_projection(text_outputs.pooler_output)

        # `image_feat` / `text_feat` 是全局级表示，后续用于语义分支和全局分支；
        # `image_tokens` / `text_tokens` 是 token 级表示，后续用于情感分支。
        image_feat = F.normalize(self.image_feature_proj(image_embed), dim=-1)
        text_feat = F.normalize(self.text_feature_proj(text_embed), dim=-1)

        if self.use_interactive_encoding:
            # 在 LoRA 更新前先做一次轻量图文交互，让后续表示更早感知跨模态冲突。
            image_feat = self.image_lora(image_feat + 0.5 * text_feat)
            text_feat = self.text_lora(text_feat + 0.5 * image_feat)

        image_tokens = self.image_token_proj(vision_outputs.last_hidden_state)
        text_tokens = self.text_token_proj(text_outputs.last_hidden_state)

        return {
            "image_feat": image_feat,
            "text_feat": text_feat,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
        }

    def compute_incongruity(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算语义不一致性。

        核心思想：
        - 先构造多种图文关系视角；
        - 再用卷积压缩成相关响应；
        - 最后用方差权重突出差异更明显的维度。
        """
        pair_stack = torch.stack(
            [
                image_feat,
                text_feat,
                torch.abs(image_feat - text_feat),
                image_feat * text_feat,
            ],
            dim=1,
        )
        correlation_map = self.correlation_conv(pair_stack).squeeze(1)

        variance_weight = torch.var(torch.stack([image_feat, text_feat], dim=1), dim=1, unbiased=False)
        variance_weight = variance_weight / variance_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        sem_vector = torch.abs(correlation_map) * variance_weight
        sem_gap = sem_vector.sum(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(image_feat, text_feat, dim=-1).unsqueeze(-1)
        cos_gap = 1.0 - cos_sim
        sem_gap = sem_gap + cos_gap
        semantic_repr = self.semantic_proj(torch.cat([image_feat, text_feat, sem_vector], dim=-1))

        return {
            "semantic_repr": semantic_repr,
            "sem_gap": sem_gap,
            "sem_vector": sem_vector,
            "correlation_map": correlation_map,
            "variance_weight": variance_weight,
            "cos_sim": cos_sim,
            "cos_gap": cos_gap,
        }

    def compute_affective(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算情感不一致性。

        核心思想：
        - 先通过跨模态注意力交换上下文；
        - 再对图像和文本分别估计情感分数；
        - 最后用两者的绝对差作为 affective gap。
        """
        text_padding_mask = ~attention_mask.bool()

        # 先通过跨模态注意力交换上下文，再计算情感差值，使情感不一致性
        # 建立在“看过对方信息后”的表示之上。
        image_context, image_attn = self.image_to_text_attn(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        text_context, text_attn = self.text_to_image_attn(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        image_context_feat = image_context.mean(dim=1)
        text_context_feat = text_context.mean(dim=1)

        image_affective_embed = self.image_affective_encoder(torch.cat([image_feat, image_context_feat], dim=-1))
        text_affective_embed = self.text_affective_encoder(torch.cat([text_feat, text_context_feat], dim=-1))

        image_sentiment_embed = self.dropout(self.sentiment_relu(self.sentiment_fc1(image_affective_embed)))
        text_sentiment_embed = self.dropout(self.sentiment_relu(self.sentiment_fc1(text_affective_embed)))

        image_affect = self.sentiment_fc2(image_sentiment_embed)
        text_affect = self.sentiment_fc2(text_sentiment_embed)

        aff_gap_vector = torch.abs(image_sentiment_embed - text_sentiment_embed)
        aff_gap = aff_gap_vector.mean(dim=-1, keepdim=True)

        # ============================================
        # Continuous Contrastive Learning（DIP 式）
        # 关键改进：
        # 1) 移除 torch.no_grad()，允许梯度通过极性差异回传
        # 2) 使用高斯核软标签（连续监督）
        # 3) 使用对称 KL（双向）替代单向 KL，训练更稳定
        # ============================================
        polarity_diff = torch.abs(image_affect - text_affect.t())  # [B, B]
        tau = 0.2
        contrast_label = torch.exp(-polarity_diff / tau)
        contrast_label = contrast_label / contrast_label.sum(dim=1, keepdim=True).clamp_min(1e-6)

        temp = 0.1
        sim_matrix = torch.mm(
            F.normalize(image_sentiment_embed, dim=1),
            F.normalize(text_sentiment_embed, dim=1).t(),
        ) / temp
        log_sim = F.log_softmax(sim_matrix, dim=-1)

        kl_forward = F.kl_div(log_sim, contrast_label, reduction="batchmean", log_target=False)
        kl_backward = F.kl_div(contrast_label.clamp_min(1e-8).log(), log_sim.exp(), reduction="batchmean", log_target=False)
        sentiment_contrast_loss = 0.5 * (kl_forward + kl_backward)

        # ============================================
        # Batch Hard Contrastive Loss（可选）
        # 讽刺样本：强制图文情感嵌入距离更大
        # 非讽刺样本：强制图文情感嵌入距离更小
        # ============================================
        hard_contrastive_loss = torch.tensor(0.0, device=image_feat.device)
        if labels is not None and self.training:
            dist_matrix = torch.cdist(image_sentiment_embed, text_sentiment_embed, p=2)
            pos_dist = torch.diag(dist_matrix)

            sarcastic_mask = labels.float()
            non_sarcastic_mask = (1 - labels).float()

            margin_pos = 0.5
            margin_neg = 0.2

            loss_sarcastic = sarcastic_mask * torch.clamp(margin_pos - pos_dist, min=0)
            loss_non_sarcastic = non_sarcastic_mask * torch.clamp(pos_dist - margin_neg, min=0)
            hard_contrastive_loss = (loss_sarcastic + loss_non_sarcastic).mean()

        affective_repr = self.affective_proj(torch.cat([image_context_feat, text_context_feat, aff_gap_vector], dim=-1))

        return {
            "affective_repr": affective_repr,
            "aff_gap": aff_gap,
            "image_affect": image_affect,
            "text_affect": text_affect,
            "aff_gap_vector": aff_gap_vector,
            "sentiment_contrast_loss": sentiment_contrast_loss,
            "hard_contrastive_loss": hard_contrastive_loss,
            "image_attn": image_attn,
            "text_attn": text_attn,
            "contrast_label": contrast_label.detach(),
        }

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算归一化预测熵，用于衡量当前分支的不确定性。"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        return entropy / torch.log(torch.tensor(float(logits.size(-1)), device=logits.device))

    def _fuse_logits(self, branch_logits: torch.Tensor, branch_entropies: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行 EGDF 融合。

        当 `use_entropy_guided=True` 时：
        - 熵越低，分支越自信；
        - 融合权重越大。
        """
        if self.use_entropy_guided:
            # 熵越低表示该分支越自信，因此在 EGDF 中会得到更大的融合权重。
            scaled_entropy = branch_entropies.squeeze(-1) / self.entropy_temp
            fusion_weights = F.softmax(-scaled_entropy, dim=-1)
        else:
            fusion_weights = F.softmax(self.fusion_logits, dim=0).unsqueeze(0).expand(branch_logits.size(0), -1)

        fused_logits = torch.sum(fusion_weights.unsqueeze(-1) * branch_logits, dim=1)
        fused_entropy = self._compute_entropy(fused_logits)

        return {
            "fused_logits": fused_logits,
            "fusion_weights": fusion_weights,
            "fused_entropy": fused_entropy,
        }

    def _build_memory_features(
        self,
        sem_gap: torch.Tensor,
        aff_gap: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """把当前样本压缩成记忆库可存储/可检索的三维签名。"""
        return torch.cat([sem_gap, aff_gap, entropy], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """模型前向流程。

        训练阶段：
        - 返回分类结果与多项损失

        测试阶段：
        - 返回分类结果、gap 信息、熵、记忆检索结果等分析信息
        """
        # ============================================================
        # 1. 提取基础特征
        # ============================================================
        features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        # ============================================================
        # 2. 分别计算语义不一致性和情感不一致性
        # ============================================================
        incongruity = self.compute_incongruity(features["image_feat"], features["text_feat"])
        affective = self.compute_affective(
            image_tokens=features["image_tokens"],
            text_tokens=features["text_tokens"],
            image_feat=features["image_feat"],
            text_feat=features["text_feat"],
            attention_mask=attention_mask,
            labels=labels,
        )

        if not self.use_affective_gap:
            # 消融实验中直接关闭情感分支，但保持张量形状不变，
            # 这样后续融合与损失计算代码都不需要改动。
            affective["affective_repr"] = torch.zeros_like(affective["affective_repr"])
            affective["aff_gap"] = torch.zeros_like(affective["aff_gap"])
            affective["image_affect"] = torch.zeros_like(affective["image_affect"])
            affective["text_affect"] = torch.zeros_like(affective["text_affect"])
            affective["aff_gap_vector"] = torch.zeros_like(affective["aff_gap_vector"])
            affective["sentiment_contrast_loss"] = torch.zeros((), device=features["image_feat"].device)
            affective["hard_contrastive_loss"] = torch.zeros((), device=features["image_feat"].device)
            batch_size = features["image_feat"].size(0)
            affective["contrast_label"] = torch.zeros((batch_size, batch_size), device=features["image_feat"].device)

        # ============================================================
        # 3. 构建全局分支表示
        # ============================================================
        abs_diff = torch.abs(features["image_feat"] - features["text_feat"])
        interaction = features["image_feat"] * features["text_feat"]
        global_repr = self.global_proj(
            torch.cat([features["image_feat"], features["text_feat"], abs_diff, interaction], dim=-1)
        )

        # ============================================================
        # 4. 三个分支分别输出 logits
        # ============================================================
        semantic_logits = self.semantic_head(incongruity["semantic_repr"])
        affective_logits = self.affective_head(affective["affective_repr"])
        global_logits = self.global_head(global_repr)

        branch_logits = torch.stack([semantic_logits, affective_logits, global_logits], dim=1)
        branch_entropies = torch.stack(
            [
                self._compute_entropy(semantic_logits),
                self._compute_entropy(affective_logits),
                self._compute_entropy(global_logits),
            ],
            dim=1,
        )

        # 最终融合三条决策流：语义不一致性、情感不一致性和全局分类分支。
        # ============================================================
        # 5. EGDF 融合
        # ============================================================
        fusion_info = self._fuse_logits(branch_logits, branch_entropies)
        fused_logits = fusion_info["fused_logits"]
        fused_probs = F.softmax(fused_logits, dim=-1)

        memory_features = self._build_memory_features(
            sem_gap=incongruity["sem_gap"],
            aff_gap=affective["aff_gap"],
            entropy=fusion_info["fused_entropy"],
        )

        # ============================================================
        # 6. 测试时记忆增强
        # ============================================================
        memory_info: Optional[Dict[str, torch.Tensor]] = None
        if self.use_test_memory and not self.training:
            # 测试阶段用记忆库检索结果作为额外先验，轻微拉动最终 logits，
            # 让预测更贴近历史上见过的相似不一致模式。
            memory_info = self.test_memory.query(memory_features)
            fused_logits = fused_logits + self.memory_logit_scale * memory_info["normalized_scores"]
            fused_probs = F.softmax(fused_logits, dim=-1)

            if update_memory:
                pseudo_labels = fused_probs.argmax(dim=-1)
                self.test_memory.update(memory_features, pseudo_labels, fusion_info["fused_entropy"])
        elif self.use_test_memory and update_memory and labels is not None:
            self.test_memory.update(memory_features, labels, fusion_info["fused_entropy"])

        # ============================================================
        # 7. 整理通用输出
        # ============================================================
        predictions = fused_probs.argmax(dim=-1)
        output = {
            "logits": fused_logits,
            "probabilities": fused_probs,
            "predictions": predictions,
            "semantic_logits": semantic_logits,
            "affective_logits": affective_logits,
            "global_logits": global_logits,
            "fusion_weights": fusion_info["fusion_weights"],
            "branch_entropies": branch_entropies.squeeze(-1),
            "entropy": fusion_info["fused_entropy"],
            "sem_gap": incongruity["sem_gap"],
            "aff_gap": affective["aff_gap"],
            "memory_features": memory_features,
            "correlation_map": incongruity["correlation_map"],
            "variance_weight": incongruity["variance_weight"],
            "image_affect": affective["image_affect"],
            "text_affect": affective["text_affect"],
            "sentiment_contrast_loss": affective["sentiment_contrast_loss"],
            "hard_contrastive_loss": affective["hard_contrastive_loss"],
        }

        if memory_info is not None:
            output["memory_scores"] = memory_info["normalized_scores"]
            output["memory_distances"] = memory_info["distances"]
            output["memory_counts"] = memory_info["counts"]

        # ============================================================
        # 8. 训练阶段损失
        # ============================================================
        if labels is not None and self.training:
            cls_loss = F.cross_entropy(fused_logits, labels)
            semantic_loss = F.cross_entropy(semantic_logits, labels)
            affective_loss = F.cross_entropy(affective_logits, labels)
            global_loss = F.cross_entropy(global_logits, labels)
            branch_loss = (semantic_loss + affective_loss + global_loss) / 3.0

            # 该一致性项用于约束语义 gap 和情感 gap 不要在训练时偏离过远。
            sem_signal = torch.sigmoid(incongruity["sem_gap"])
            aff_signal = torch.sigmoid(affective["aff_gap"])
            consistency_loss = F.mse_loss(sem_signal, aff_signal)

            sentiment_contrast_loss = affective["sentiment_contrast_loss"]
            hard_contrastive_loss = affective["hard_contrastive_loss"]
            total_loss = (
                cls_loss
                + 0.3 * branch_loss
                + 0.05 * consistency_loss
                + 0.3 * sentiment_contrast_loss
                + 0.2 * hard_contrastive_loss
            )
            output.update(
                {
                    "loss": total_loss,
                    "cls_loss": cls_loss,
                    "branch_loss": branch_loss,
                    "consistency_loss": consistency_loss,
                    "sentiment_contrast_loss": sentiment_contrast_loss,
                    "hard_contrastive_loss": hard_contrastive_loss,
                }
            )

        return output
