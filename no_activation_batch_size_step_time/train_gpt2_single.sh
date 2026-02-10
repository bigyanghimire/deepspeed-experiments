#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=mega-slurm-%j.output
#SBATCH --error=mega-slurm-%j.err

# Optional: load modules or activate environment
# rm -fr *.err *.out
# module load anaconda3
module load cuda/11.8.0
# module load gcc/12.3.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 

rm -fr configs/

bash generate_baseline_config.sh 16
bash generate_bf16_config.sh 16
bash run_comparison.sh --batch_size 16
