#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu  --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=aces_ulysses_slurm_logs/standard-ulysses-%j.output
#SBATCH --error=aces_ulysses_slurm_logs/standard-ulysses-%j.err

module load CUDA/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

NUM_NODES=2
GPUS_PER_NODE=1
NUM_STEPS=20
WARMUP_STEPS=5
SEQUENCE_LENGTH=256
export PATH="${PATH}"
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="train_gpt_ulysses.py --seq_length ${SEQUENCE_LENGTH}"
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p aces_default_exp
export HF_HOME="/scratch/user/u.bg348806/.cache/huggingface"
srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}" > "aces_default_exp/default_run_${SEQUENCE_LENGTH}.log" 2>&1
# deepspeed --num_gpus 4 train_gpt_grouped.py