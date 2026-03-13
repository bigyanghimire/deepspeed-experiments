#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# #SBATCH --partition=mri2020
#SBATCH --output=mega-slurm-%j.output
#SBATCH --error=mega-slurm-%j.err

# module load cuda/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

deepspeed --num_gpus 4 train_gpt_grouped.py