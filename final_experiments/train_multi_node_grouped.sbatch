#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=ulysses_slurm_logs/standard-ulysses-%j.output
#SBATCH --error=ulysses_slurm_logs/standard-ulysses-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

NUM_NODES=2
GPUS_PER_NODE=4
NUM_STEPS=20
WARMUP_STEPS=5

# SEQUENCE_LENGTH should be passed via --export from submit script
# Default to 1024 if not set
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-1024}

LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="train_gpt_grouped.py --seq_length ${SEQUENCE_LENGTH}"
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p grouped_exp
srun -l bash -c "${LAUNCHER} ${CMD}" > "grouped_exp/grouped_run_${SEQUENCE_LENGTH}.log" 2>&1