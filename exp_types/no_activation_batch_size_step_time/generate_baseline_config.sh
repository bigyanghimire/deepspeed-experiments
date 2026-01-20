#!/usr/bin/env bash
set -euo pipefail

# ---- args ----
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <train_micro_batch_size_per_gpu>"
  exit 1
fi

TRAIN_MICRO_BATCH_SIZE="$1"

# basic sanity check
if ! [[ "$TRAIN_MICRO_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Error: train_micro_batch_size_per_gpu must be an integer"
  exit 1
fi

# ---- paths ----
CONFIG_DIR="configs"
CONFIG_FILE="${CONFIG_DIR}/baseline.json"

mkdir -p "$CONFIG_DIR"

# ---- write JSON ----
cat > "$CONFIG_FILE" <<EOF
{
    "train_micro_batch_size_per_gpu": ${TRAIN_MICRO_BATCH_SIZE},
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 0
    },

    "torch_autocast": {
        "enabled": false
    },
     "wall_clock_breakdown": true
}

EOF

echo "Wrote ${CONFIG_FILE}"
