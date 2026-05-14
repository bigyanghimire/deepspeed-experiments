#!/bin/bash
#SBATCH --job-name=deepspeed_sp
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j-%x.output
#SBATCH --error=slurm_logs/%j-%x.err

# --- Environment Setup ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Constants & Args ---
# NUM_NODES=2
# GPUS_PER_NODE=4
NUM_NODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE   # or SLURM_GPUS_PER_NODE (depends on cluster)
echo $NUM_NODES
echo $GPUS_PER_NODE
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-32768}
SEQ_PARALLEL_SIZE=$((NUM_NODES * GPUS_PER_NODE))
BATCH_SIZE=2
MODEL_NAME="meta-llama/Llama-3.2-1B"
#MODEL_NAME="meta-llama/Llama-3.2-3B"
# MODEL_NAME="Qwen/Qwen2-7B"
#MODEL_NAME="meta-llama/Llama-3.1-8B"

#MODEL_NAME="Qwen/Qwen1.5-1.8B"

#MODEL_NAME="Qwen/Qwen2.5-3B"
#MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME_SHORT="llama323"
MODE=$1  # Accept 'grouped' or 'ulysses' as first argument
EXPERIMENT_NAME="loss-plot"
# --- Logic Gate ---
if [ "$MODE" == "grouped" ]; then
    SCRIPT_NAME="../train_gpt_ulysses.py"
    LOG_DIR="experiment_logs/${EXPERIMENT_NAME}/${MODEL_NAME_SHORT}/grouped_exp"
elif [ "$MODE" == "ulysses" ]; then
    SCRIPT_NAME="../train_gpt_ulysses.py"
    LOG_DIR="experiment_logs/${EXPERIMENT_NAME}/${MODEL_NAME_SHORT}/default_exp"
else
    echo "Error: Please specify 'grouped' or 'ulysses' (e.g., sbatch train_multi_node.sh ulysses)"
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p slurm_logs

# --- Execution ---
LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="${SCRIPT_NAME} --seq_length ${SEQUENCE_LENGTH} --seq_parallel_size ${SEQ_PARALLEL_SIZE} --type ${MODE} --batch_size ${BATCH_SIZE} --model_name ${MODEL_NAME} --exp_name ${EXPERIMENT_NAME}"

echo "Running in ${MODE} mode using ${SCRIPT_NAME}..."

srun --export=ALL -l bash -c "${LAUNCHER} ${CMD}" > "${LOG_DIR}/${MODE}_${SEQUENCE_LENGTH}_batch_${BATCH_SIZE}_gpu_${SEQ_PARALLEL_SIZE}.log" 2>&1