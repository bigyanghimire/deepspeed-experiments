#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=slurm_logs/grouped-ulysses-%j.output
#SBATCH --error=slurm_logs/grouped-ulysses-%j.err

# Optional: load modules or activate environment
# rm -fr *.err *.out
# module load anaconda3

# module load gcc/12.3.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
set -e
cd "$SLURM_SUBMIT_DIR"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

NUM_NODES=2
GPUS_PER_NODE=4
NUM_LAYERS=32
HIDDEN_DIM=2048
BATCH_SIZE=1
SEQ_LENGTH=512
NUM_STEPS=100
WARMUP_STEPS=5
MASTER_PORT=29501
COMMON_ARGS="--num_layers $NUM_LAYERS --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS --activation_checkpointing --use_real_data --seed 42"

LAUNCHER="torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

CMD="gpt_model.py --deepspeed_config configs/baseline.json $COMMON_ARGS"
export HF_TOKEN=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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
srun -l bash -c "${LAUNCHER} ${CMD}"