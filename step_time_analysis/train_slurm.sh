#!/bin/bash
#SBATCH --job-name=train_bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
##SBATCH --partition=mri2020
#SBATCH --output=mega-slurm-%j.output
#SBATCH --error=mega-slurm-%j.err

# Optional: load modules or activate environment
# rm -fr *.err *.out
# module load anaconda3
module load cuda/11.8.0
# module load gcc/12.3.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate ds-hf 
# source activate cuda-python
# export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
# export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# export PATH=$CONDA_PREFIX/bin:$PATH

# which deepspeed || echo "deepspeed missing in train.sh"

# PYTHONPATH=$PYTHONPATH:./megatron torchrun --nproc-per-node 2 examples/run_simple_mcore_train_loop.py
# deepspeed --num_gpus=2 train3.py --deepspeed_config configs/baseline.json \
#   --num_layers 32 --hidden_dim 4096 --num_heads 32 --batch_size 1 \
#   --num_steps 1000 --activation_checkpointing \
#   --loss_log_file logs/baseline_loss.csv --use_real_data --seed 42


# deepspeed --num_gpus=4 train.py --deepspeed_config configs/bf16_full.json \
#   --num_layers 32 --hidden_dim 4096 --num_heads 32 --batch_size 1 \
#   --num_steps 1000 --activation_checkpointing \
#   --loss_log_file logs/bf16_full_loss.csv --use_real_data --seed 42
#bash run_comparison.sh
bash run_profiler.sh