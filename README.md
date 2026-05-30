# CAIP（CLIP-based Adaptive Incongruity Perception）

一个用于多模态讽刺检测的 PyTorch 项目。模型冻结 CLIP backbone，在其上构建三条决策分支并动态融合：
- **语义不一致性分支**（semantic incongruity）
- **情感不一致性分支**（affective incongruity）
- **全局交互分支**（global interaction）

融合采用 **EGDF（Entropy-Guided Dynamic Fusion）**：按样本、按分支预测熵动态分配权重。评估阶段可选启用 **测试时记忆库**（Test-Time Memory）做顺序检索增强。

本仓库只包含代码与脚本：**数据集与训练产物不上传**（已通过 `.gitignore` 排除 `MMSD* / data/processed / runs / checkpoints / *.pt / *.pkl / *.png` 等）。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA 11.8（推荐安装方式）

如果你的机器是 CUDA 11.8，建议不要直接依赖 `requirements.txt` 里的 `torch>=...` 去自动解析（可能会装到 CPU 版）。
更稳的方式是先安装 cu118 的 PyTorch wheel，再安装其余依赖：

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
pip install -r requirements.txt
```

验证是否成功启用 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 数据格式（JSONL）

训练/评估输入为 JSONL，每行一个样本：

```json
{"image_path": "path/to/image.jpg", "text": "sample text", "label": 1}
```

- `label`: 0=非讽刺，1=讽刺

## 数据准备（示例：MMSD2.0）

```bash
python prepare_data.py \
  --raw-root /path/to/MMSD2.0 \
  --output-dir data/processed
```

输出：
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

### 数据集目录说明（不推送到 GitHub）

你本地可能会有如下目录（例如只包含 `train.json/valid.json/test.json` 的划分文件，或包含完整图片与文本的原始数据）：
- `MMSD1.0/`
- `MMSD2.0/`

这些都属于数据集内容，本仓库默认不提交它们（`.gitignore` 已忽略）。同样，`data/processed/` 下生成的 JSONL 也默认不提交。

建议：
- 将原始数据与处理后的 JSONL 都留在本地/服务器；
- 只把代码（`src/` 与各入口脚本）推送到 GitHub。

## 训练（train.py）

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

说明：
- 若希望测试时记忆库“边测边写”，评估阶段需要 `batch_size=1`（顺序更新）。

## 产物位置（只看最新的这一套）

最新流程的所有训练/评估产物都在 `runs/` 目录下：

```text
runs/
├── train/<experiment>/<run_id>/
│   ├── history.json
│   ├── metrics.json
│   ├── analysis_cache.pkl
│   ├── memory_stats.png
│   ├── run_manifest.json
│   └── checkpoints/best_model.pt
└── eval/<experiment>/<run_id>/
    ├── metrics.json
    ├── analysis_cache.pkl
    ├── memory_stats.png
    └── run_manifest.json
```

（仓库里可能存在 `logs/` 等历史目录，但它们不属于当前默认流程；当前默认只看 `runs/`。）

## 可视化（analyze.py）

```bash
python analyze.py \
  --cache-path runs/train/<experiment>/<run_id>/analysis_cache.pkl \
  --output-path runs/train/<experiment>/<run_id>/caip_analysis.png
```

## 代码入口（从这里开始读）

- 模型与记忆库：[src/models/caip_model.py](src/models/caip_model.py)
- 训练/评估与指标：[src/training/trainer.py](src/training/trainer.py)
- ECE 指标：[src/utils/metrics.py](src/utils/metrics.py)