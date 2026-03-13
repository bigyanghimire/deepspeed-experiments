#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=mega-slurm-%j.output
#SBATCH --error=mega-slurm-%j.err

module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
set -e
nvidia-smi
cd "$SLURM_SUBMIT_DIR"

NUM_NODES=2
GPUS_PER_NODE=2
NUM_LAYERS=32
HIDDEN_DIM=2048
BATCH_SIZE=16
SEQ_LENGTH=1024
NUM_STEPS=100
WARMUP_STEPS=5
MASTER_PORT=29600

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Set up master address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
echo "=============================================="
echo "BF16 Low-Precision Master Weights Memory Test"
echo "=============================================="
echo "Configuration:"
echo "  NUM_NODES: $NUM_NODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo "  NUM_LAYERS: $NUM_LAYERS"
echo "  HIDDEN_DIM: $HIDDEN_DIM"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  SEQ_LENGTH: $SEQ_LENGTH"
echo "  NUM_STEPS: $NUM_STEPS"
echo "  LOG_DIR: $LOG_DIR"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "=============================================="

COMMON_ARGS="--num_layers $NUM_LAYERS --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS --activation_checkpointing --use_real_data --seed 42"

# Define torchrun launcher
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# Training command
CMD="train2.py \
    --deepspeed_config configs/baseline.json \
    --loss_log_file logs/baseline_loss.csv \
    ${COMMON_ARGS}"

# Training command
CMD2="train2.py \
    --deepspeed_config configs/bf16_full.json \
    --loss_log_file logs/bf16full_loss.csv \
    ${COMMON_ARGS}"

echo ""
echo "[1/2] Running BASELINE (bf16 with fp32 master weights/grads/optimizer states)..."
echo "----------------------------------------------"

# Execute with srun
srun -l bash -c "${LAUNCHER} ${CMD}" 2>&1 | tee "$LOG_DIR/baseline.log"
# srun -l bash -c "${LAUNCHER} ${CMD2}" 2>&1 | tee "$LOG_DIR/bf16_full.log"