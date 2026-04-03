#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=grouped_slurm_logs/grouped-%j.output
#SBATCH --error=grouped_slurm_logs/grouped-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
nvidia-smi
NUM_NODES=2
GPUS_PER_NODE=4
NUM_STEPS=20
WARMUP_STEPS=5

# SEQUENCE_LENGTH should be passed via --export from submit script
# Default to 1024 if not set
SEQUENCE_LENGTH=32000

LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
    
SEQ_PARALLEL_SIZE=$((NUM_NODES * GPUS_PER_NODE))
CMD="train_gpt_ulysses.py --seq_length ${SEQUENCE_LENGTH} --seq_parallel_size ${SEQ_PARALLEL_SIZE}"
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p grouped_exp
srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}"