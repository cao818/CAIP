# CAIP Project

PyTorch implementation scaffold for **CAIP: CLIP-based Adaptive Incongruity Perception** on multimodal sarcasm detection.

## Features

- CLIP backbone with frozen encoder and LoRA-based interactive adaptation
- Explicit semantic and affective incongruity modeling
- Entropy-Guided Dynamic Fusion (EGDF)
- Test-Time Memory for sequential inference
- Training, evaluation, visualization, and batch experiment scripts

## Project Structure

```text
CAIP_Project/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   └── utils/
├── data/processed/
├── checkpoints/
├── logs/
├── experiments/
├── train.py
├── eval.py
├── analyze.py
├── prepare_data.py
├── run_experiments.sh
└── requirements.txt
```

## Installation

```bash
cd /root/work/CAIP_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

The training and evaluation scripts expect JSONL files with the following schema:

```json
{"image_path": "/absolute/or/relative/path/to/image.jpg", "text": "sample text", "label": 1}
```

- `image_path`: path to image file
- `text`: associated text
- `label`: `0` for non-sarcastic, `1` for sarcastic

## Prepare MMSD2.0

Assumed raw structure:

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

Run:

```bash
python prepare_data.py \
  --raw-root /path/to/MMSD2.0 \
  --output-dir /root/work/CAIP_Project/data/processed
```

Output:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

## Training

Basic training command:

```bash
python train.py \
  --train_file data/processed/train.jsonl \
  --val_file data/processed/val.jsonl \
  --test_file data/processed/test.jsonl \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --lora_rank 8 \
  --entropy_temp 1.0 \
  --memory_size 512 \
  --ablation none \
  --patience 3
```

If your server has restricted network access, you can also provide a local CLIP
directory or a Hugging Face token:

```bash
python train.py \
  --train_file data/processed/train.jsonl \
  --val_file data/processed/val.jsonl \
  --test_file data/processed/test.jsonl \
  --clip_model_name /path/to/local/clip-vit-base-patch32 \
  --local_files_only
```

Key outputs:

- best checkpoint: `checkpoints/best_model.pt`
- history log: `logs/train_history.json`
- test metrics: `logs/test_metrics.json`
- analysis cache: `analysis_cache.pkl`
- memory plot: `memory_stats.png`

## Evaluation

Evaluate a trained checkpoint without retraining:

```bash
python eval.py \
  --test_file data/processed/test.jsonl \
  --checkpoint checkpoints/best_model.pt \
  --lora_rank 8 \
  --entropy_temp 1.0 \
  --memory_size 512 \
  --ablation none
```

Key outputs:

- evaluation metrics JSON
- `analysis_cache.pkl`
- `memory_stats.png`

## Visualization

Generate the main CAIP analysis figure:

```bash
python analyze.py \
  --cache-path analysis_cache.pkl \
  --output-path caip_analysis.png
```

Generate memory bank statistics:

```python
from src.utils import plot_memory_stats

plot_memory_stats(memory_source, output_path="memory_stats.png")
```

## Batch Experiments

Run the full experiment suite:

```bash
bash run_experiments.sh
```

This script runs:

- baseline
- full CAIP
- 4 ablations
- LoRA rank sensitivity experiments

It finally aggregates F1 scores into:

- `experiments/f1_summary.tsv`

Each experiment also saves:

- `logs/experiments/<name>.log`
- `logs/experiments/<name>_history.json`
- `logs/experiments/<name>_analysis_cache.pkl`
- `logs/experiments/<name>_memory_stats.png`
- `logs/experiments/metrics/<name>.json`

## Main Arguments

### `train.py`

- `--train_file`, `--val_file`, `--test_file`
- `--lora_rank`, `--entropy_temp`, `--memory_size`
- `--ablation [none|no_interactive|no_entropy|no_memory|no_affective]`
- `--epochs`, `--batch_size`, `--lr`
- `--patience`
- `--hf_token`, `--hf_cache_dir`, `--local_files_only`

### `eval.py`

- `--test_file`
- `--checkpoint`
- `--lora_rank`, `--entropy_temp`, `--memory_size`
- `--ablation`
- `--metrics_output`, `--analysis_cache`, `--memory_plot`
- `--hf_token`, `--hf_cache_dir`, `--local_files_only`

## Metrics

Current evaluation pipeline reports:

- Accuracy
- F1
- Precision
- Recall
- ECE (Expected Calibration Error)
- Easy accuracy
- Hard accuracy

## Reproducibility

- Automatic CUDA detection with CPU fallback
- Fixed random seed via `torch.manual_seed(42)` and aligned `numpy/random/cuda` seeds
- Early stopping enabled with default `patience=3`

## Notes

- The CLIP backbone is loaded from Hugging Face and may require internet access on first run.
- The code disables the Hugging Face `xet` download path by default to avoid common 401/CAS issues on restricted servers.
- Test-Time Memory is only updated sequentially during `batch_size=1` evaluation.
- Corrupted images are skipped with warnings instead of crashing the dataloader.
