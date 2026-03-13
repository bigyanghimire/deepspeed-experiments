#!/bin/bash
#SBATCH --job-name=profile_train
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=mri2020

module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf
set -e

cd "$SLURM_SUBMIT_DIR"

# NCCL debugging and NVTX markers
export NCCL_DEBUG=INFO
export NCCL_NVTX_TRACE=1

NUM_NODES=2
GPUS_PER_NODE=1
NUM_STEPS=20
WARMUP_STEPS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="profiles/${TIMESTAMP}"
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$PROFILE_DIR" "$LOG_DIR"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29600
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Profiling Configuration:"
echo "  NUM_NODES: $NUM_NODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  NUM_STEPS: $NUM_STEPS"
echo "  PROFILE_DIR: $PROFILE_DIR"
echo "  NCCL_NVTX_TRACE: $NCCL_NVTX_TRACE"
echo "  Nsys version: $(nsys --version)"
echo "=============================================="

COMMON_ARGS="--num_layers 32 --hidden_dim 2048 --batch_size 8 --seq_length 512 --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS --activation_checkpointing --use_real_data --seed 42"

LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="train2.py \
    --deepspeed_config configs/baseline.json \
    --loss_log_file logs/baseline_loss.csv \
    ${COMMON_ARGS}"

# Nsys command without nccl trace
NSYS_CMD="nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=${PROFILE_DIR}/profile_rank\${SLURM_PROCID} \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --stats=true \
    --cuda-memory-usage=true"

echo "Starting profiling run (rank 0 only)..."

srun -l bash -c "
    if [ \$SLURM_PROCID -eq 0 ]; then
        echo '[Rank 0] Profiling enabled'
        ${NSYS_CMD} ${LAUNCHER} ${CMD}
    else
        echo '[Rank '\$SLURM_PROCID'] Running without profiling'
        ${LAUNCHER} ${CMD}
    fi
" 2>&1 | tee "$LOG_DIR/run.log"

echo ""
echo "=============================================="
echo "Profiling complete!"
echo "Profile location: $PROFILE_DIR"
echo "To download: scp -r $USER@cluster:$PWD/$PROFILE_DIR ."
echo "To view: nsys-ui $PROFILE_DIR/profile_rank0.nsys-rep"
echo "=============================================="