#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/train.py}"

DATASET_NAME="${DATASET_NAME:-MMSD2.0}"
DEFAULT_DATA_DIR="$ROOT_DIR/data/processed/$DATASET_NAME"
if [[ -d "$DEFAULT_DATA_DIR" ]]; then
  DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
else
  DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/processed}"
fi
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DATA_DIR/val.jsonl}"
TEST_JSONL="${TEST_JSONL:-$DATA_DIR/test.jsonl}"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/batch}"
SUITE_ID="${SUITE_ID:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="$RUN_ROOT/$SUITE_ID"
LOG_DIR="$SUITE_DIR/console_logs"
SUMMARY_PATH="${SUMMARY_PATH:-$SUITE_DIR/f1_summary.tsv}"

mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_PATH")"

EXTRA_ARGS=("$@")

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing ${label}: $path" >&2
    exit 1
  fi
}

require_file "$TRAIN_SCRIPT" "training script"
require_file "$TRAIN_JSONL" "train jsonl"
require_file "$VAL_JSONL" "val jsonl"
require_file "$TEST_JSONL" "test jsonl"

run_experiment() {
  local name="$1"
  local use_interactive="$2"
  local use_entropy="$3"
  local use_memory="$4"
  local use_affective="$5"
  local lora_rank="$6"

  local log_file="$LOG_DIR/${name}.log"

  local cmd=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --experiment-name "$name"
    --train-jsonl "$TRAIN_JSONL"
    --val-jsonl "$VAL_JSONL"
    --test-jsonl "$TEST_JSONL"
    --run-root "$SUITE_DIR"
    --run-id "train"
    --use-interactive-encoding "$use_interactive"
    --use-entropy-guided "$use_entropy"
    --use-test-memory "$use_memory"
    --use-affective-gap "$use_affective"
    --lora-rank "$lora_rank"
  )

  if ((${#EXTRA_ARGS[@]} > 0)); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[RUN] $name"
  echo "[CMD] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log_file"
  echo "[DONE] $name"
}

aggregate_f1_scores() {
  "$PYTHON_BIN" - "$SUITE_DIR" "$LOG_DIR" "$SUMMARY_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

suite_dir = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

patterns = [
    re.compile(r'"?f1"?\s*[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE),
    re.compile(r'\bf1-score\b\s*[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE),
]


def extract_from_json(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        for key in ("f1", "test_f1", "val_f1", "best_f1"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def extract_from_log(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    matches = []
    for pattern in patterns:
        matches.extend(pattern.findall(text))
    if not matches:
        return None
    return float(matches[-1])


results = {}
for metrics_path in sorted(suite_dir.glob("**/metrics.json")):
    value = extract_from_json(metrics_path)
    if value is not None:
        name = None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                name = payload.get("experiment_name")
        except Exception:
            name = None
        if not isinstance(name, str) or not name.strip():
            name = metrics_path.parent.parent.name
        results[name] = value

for log_path in sorted(log_dir.glob("*.log")):
    if log_path.stem not in results:
        value = extract_from_log(log_path)
        if value is not None:
            results[log_path.stem] = value

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as file:
    file.write("experiment\tf1\n")
    for name, value in sorted(results.items()):
        file.write(f"{name}\t{value:.6f}\n")

print("F1 summary")
for name, value in sorted(results.items()):
    print(f"{name}\t{value:.6f}")
print(f"Saved summary to: {summary_path}")
PY
}

echo "[INFO] Starting experiment suite"
echo "[INFO] Suite dir: $SUITE_DIR"
echo "[INFO] Data dir: $DATA_DIR"

# 1. Baseline (all CAIP modules disabled)
run_experiment "baseline" 0 0 0 0 8

# 2. CAIP full model
run_experiment "caip_full" 1 1 1 1 8

# 3. Ablation studies
run_experiment "no_interactive" 0 1 1 1 8
run_experiment "no_entropy" 1 0 1 1 8
run_experiment "no_memory" 1 1 0 1 8
run_experiment "no_affective" 1 1 1 0 8

# 4. LoRA rank sensitivity
run_experiment "lora_rank_4" 1 1 1 1 4
run_experiment "lora_rank_8" 1 1 1 1 8
run_experiment "lora_rank_16" 1 1 1 1 16
run_experiment "lora_rank_32" 1 1 1 1 32

aggregate_f1_scores
