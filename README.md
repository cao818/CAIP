# CAIP：基于 CLIP 的多模态讽刺检测（Adaptive Incongruity Perception）

本项目是一个 PyTorch 实现，用于多模态讽刺检测任务。模型以冻结的 CLIP 作为 backbone，在其上构建：
- 语义不一致性分支（Semantic Incongruity）
- 情感不一致性分支（Affective Incongruity）
- 全局交互分支（Global Interaction）

三条分支通过 EGDF（Entropy-Guided Dynamic Fusion）按样本动态融合，并在评估阶段可选启用测试时记忆库（Test-Time Memory）做顺序检索增强。

数据集与训练产物不会随仓库发布（已通过 `.gitignore` 排除）。

## 特性

- 冻结 CLIP backbone，减少训练开销
- LoRAInjector：轻量交互式适配（不微调 CLIP 主干）
- 三分支不一致性建模：semantic / affective / global
- EGDF 熵引导动态融合：按分支预测熵分配融合权重
- AdvancedTestTimeMemory：测试时双通道记忆库（0=非讽刺，1=讽刺），支持熵排序替换与可选分布先验
- 统一训练与评估脚本，自动保存 run 目录（history/metrics/cache/manifest）

## 项目结构

```text
CAIP_Project/
├── src/
│   ├── models/            # CAIPModel、LoRAInjector、AdvancedTestTimeMemory
│   ├── data/              # Dataset & dataloader
│   ├── training/          # Trainer（训练/评估/指标）
│   └── utils/             # 运行配置、ECE、可视化
├── train.py               # 训练入口（含测试与缓存导出）
├── eval.py                # 评估入口（加载 checkpoint 跑 test）
├── analyze.py             # 可视化 analysis_cache.pkl
├── prepare_data.py        # 数据预处理（生成 JSONL）
├── run_experiments.sh     # 批量实验脚本
├── requirements.txt
└── README.md
```

## 环境与安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你的环境无法从 Hugging Face 下载模型，建议：
- 使用 `--local_files_only` + 本地 CLIP 权重目录
- 或设置 `--hf_token`（私有/限流场景）

## 数据格式（JSONL）

训练/评估脚本期望 JSONL，每行一个样本：

```json
{"image_path": "path/to/image.jpg", "text": "sample text", "label": 1}
```

- `image_path`：图像路径（相对/绝对均可）
- `text`：文本
- `label`：0=非讽刺，1=讽刺

## 数据准备（以 MMSD2.0 为例）

假设原始结构：

```text
MMSD2.0/
├── train/
│   ├── image/
│   ├── text.txt
│   └── label.txt
├── val/
│   ├── image/
│   ├── text.txt
│   └── label.txt
└── test/
    ├── image/
    ├── text.txt
    └── label.txt
```

生成 JSONL：

```bash
python prepare_data.py \
  --raw-root /path/to/MMSD2.0 \
  --output-dir data/processed
```

输出：
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

## 训练（train.py）

最小训练命令：

```bash
python train.py \
  --train_file data/processed/train.jsonl \
  --val_file data/processed/val.jsonl \
  --test_file data/processed/test.jsonl \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --lora_rank 16 \
  --entropy_temp 1.0 \
  --memory_size 512 \
  --ablation none \
  --patience 8 \
  --seed 42
```

离线/受限网络（使用本地 CLIP）：

```bash
python train.py \
  --train_file data/processed/train.jsonl \
  --val_file data/processed/val.jsonl \
  --test_file data/processed/test.jsonl \
  --clip_model_name /path/to/local/clip-vit-base-patch32 \
  --local_files_only
```

## 评估（eval.py）

只评估 checkpoint（不重新训练）：

```bash
python eval.py \
  --test_file data/processed/test.jsonl \
  --checkpoint runs/train/<experiment>/<run_id>/checkpoints/best_model.pt \
  --batch_size 1 \
  --lora_rank 16 \
  --entropy_temp 1.0 \
  --memory_size 512 \
  --ablation none
```

注意：如果你希望测试时记忆库“边测边写”，评估阶段需要 `batch_size=1`（顺序更新）。

## 可视化分析（analyze.py）

`train.py` / `eval.py` 会保存 `analysis_cache.pkl`，可用下述命令生成四宫格图（entropy、alpha、gap 分布与相关性）：

```bash
python analyze.py \
  --cache-path runs/train/<experiment>/<run_id>/analysis_cache.pkl \
  --output-path runs/train/<experiment>/<run_id>/caip_analysis.png
```

## 输出产物（runs 目录）

每次训练/评估会生成一个 run 目录，并写入 `run_manifest.json` 记录配置与产物路径。常见文件：
- `history.json`：每个 epoch 的 train/val 指标
- `metrics.json`：test 指标汇总（acc/f1/p/r/ece/easy_acc/hard_acc）
- `analysis_cache.pkl`：逐样本缓存（entropy/sem_gap/aff_gap/fusion_weights/pred/label）
- `memory_stats.png`：记忆库统计图
- `checkpoints/best_model.pt`：最佳 checkpoint

## 重要实现位置（读代码入口）

- 主模型与记忆库：[src/models/caip_model.py](src/models/caip_model.py)
- 训练/评估逻辑与指标：[src/training/trainer.py](src/training/trainer.py)
- ECE 指标：[src/utils/metrics.py](src/utils/metrics.py)
- analysis_cache 可视化：[analyze.py](analyze.py)

## 常见问题

1) **为什么评估时 memory 看起来没生效？**
   - 记忆库的顺序写入依赖 `batch_size=1` 的评估，否则不会在线更新（避免 batch 乱序破坏语义）。

2) **为什么第一次运行下载 CLIP 失败？**
   - 受限网络可使用 `--local_files_only` 并提供本地 CLIP 权重目录；
   - 或设置 `--hf_token`。项目默认禁用 HF 的 `xet` 下载路径以减少 401/CAS 问题。
