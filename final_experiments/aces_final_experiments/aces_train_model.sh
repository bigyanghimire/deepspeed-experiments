#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu  --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%j-%x.output
#SBATCH --error=slurm_logs/%j-%x.err

module load CUDA/11.8.0
module load GCC/9.3.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/scratch/user/u.bg348806/.cache/huggingface/hub"
export HF_HUB_CACHE="/scratch/user/u.bg348806/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/user/u.bg348806/.cache/huggingface/datasets"
export PATH="${PATH}"

NUM_NODES=2
GPUS_PER_NODE=1
SEQUENCE_LENGTH=256
SEQ_PARALLEL_SIZE=$((NUM_NODES * GPUS_PER_NODE))

MODE=$1  # Accept 'grouped' or 'ulysses' as first argument

# --- Logic Gate ---
if [ "$MODE" == "grouped" ]; then
    SCRIPT_NAME="../train_gpt_ulysses.py"
    LOG_DIR="grouped_exp"
elif [ "$MODE" == "ulysses" ]; then
    SCRIPT_NAME="../train_gpt_ulysses.py"
    LOG_DIR="default_exp"
else
    echo "Error: Please specify 'grouped' or 'ulysses' (e.g., sbatch train_multi_node.sh ulysses)"
    exit 1
fi

mkdir -p $LOG_DIR
mkdir -p slurm_logs

LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"


CMD="${SCRIPT_NAME} --seq_length ${SEQUENCE_LENGTH} --seq_parallel_size ${SEQ_PARALLEL_SIZE} --type ${MODE}"

echo "Running in ${MODE} mode using ${SCRIPT_NAME}..."

srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}" > "${LOG_DIR}/${MODE}_run_${SEQUENCE_LENGTH}.log" 2>&1
# deepspeed --num_gpus 4 train_gpt_grouped.py

