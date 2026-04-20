#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=benchmark_logs/benchmarks-%j.output
#SBATCH --error=benchmark_logs/benchmarks-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

NUM_NODES=1
GPUS_PER_NODE=2
NUM_STEPS=20
WARMUP_STEPS=5
SEQUENCE_LENGTH=32000
export PATH="${PATH}"
OUTPUT_FILE="alltoall_benchmark_${SLURM_JOB_ID}.json"
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="benchmark.py --output=${OUTPUT_FILE}"

srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}"
# deepspeed --num_gpus 4 train_gpt_grouped.py