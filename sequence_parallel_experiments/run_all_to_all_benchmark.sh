#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=mega-slurm-%j.output
#SBATCH --error=mega-slurm-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
set -e
nvidia-smi
cd "$SLURM_SUBMIT_DIR"

NUM_NODES=2
GPUS_PER_NODE=4
NUM_LAYERS=32
HIDDEN_DIM=2048
BATCH_SIZE=16
SEQ_LENGTH=1024
NUM_STEPS=100
WARMUP_STEPS=5
MASTER_PORT=29600


# Set up master address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT

# Define torchrun launcher
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# Training command
# CMD="simple_all_to_all.py"
# CMD="distributed_benchmarks.py"
CMD="distributed_benchmarks_multinode.py"

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# Execute with srun
srun -l bash -c "${LAUNCHER} ${CMD}"
# export NCCL_SOCKET_IFNAME="^docker0,lo"
# export NCCL_IB_DISABLE=1
# deepspeed --num_gpus 4 distributed_benchmarks.py
# srun -l bash -c "${LAUNCHER} ${CMD2}" 2>&1 | tee "$LOG_DIR/bf16_full.log"