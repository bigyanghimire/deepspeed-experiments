#!/bin/bash
#SBATCH --job-name=setupenv
#SBATCH --nodes=1
#SBATCH --partition=gpu  --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=ulysses_slurm_logs/standard-ulysses-%j.output
#SBATCH --error=ulysses_slurm_logs/standard-ulysses-%j.err

module load CUDA/11.8.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
micromamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia