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
PROJECT_DIR="$SCRATCH/bigyan_project"
MICROMAMBA_DIR="$PROJECT_DIR/.local/bin"
eval "$("$MICROMAMBA_DIR/micromamba" shell hook -s bash)"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf2 

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
set -a          # auto-export variables
source ../../.env
set +a
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${PROJECT_DIR}/.cache/huggingface/hub"
export HF_HUB_CACHE="${PROJECT_DIR}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${PROJECT_DIR}/.cache/huggingface/datasets"
export PATH="${PATH}"

NUM_NODES=2
GPUS_PER_NODE=4
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-32768}
SEQ_PARALLEL_SIZE=$((NUM_NODES * GPUS_PER_NODE))
BATCH_SIZE=2
MODEL_NAME="meta-llama/Llama-3.2-3B"
MODEL_NAME_SHORT="llama3B"
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

mkdir -p "${LOG_DIR}/${MODEL_NAME_SHORT}"
mkdir -p slurm_logs

# --- Execution ---
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="${SCRIPT_NAME} --seq_length ${SEQUENCE_LENGTH} --seq_parallel_size ${SEQ_PARALLEL_SIZE} --type ${MODE} --batch_size ${BATCH_SIZE} --model_name ${MODEL_NAME}"

echo "Running in ${MODE} mode using ${SCRIPT_NAME}..."

srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}" > "${LOG_DIR}/${MODEL_NAME_SHORT}/${MODE}_${SEQUENCE_LENGTH}_batch_${BATCH_SIZE}.log" 2>&1