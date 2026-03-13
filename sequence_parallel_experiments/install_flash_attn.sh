#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=flash_attn-%j.output
#SBATCH --error=flash_attn-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MAX_JOBS=32
export TORCH_CUDA_ARCH_LIST="8.0"
pip install flash-attn --no-build-isolation
# deepspeed --num_gpus 4 train_gpt_grouped.py