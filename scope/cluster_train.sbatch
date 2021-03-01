#!/bin/bash
#SBATCH --job-name=train_scope    # Job name
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:titan:1                    # Number of GPUs
#SBATCH --ntasks=1
#SBATCH --mem=30G                     # Job memory request
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=20
#SBATCH --output=/home/users/k/klug/logs/scope/train_scope_%j.log   # Standard output and error log


module load Anaconda3

cd /home/users/k/klug/scope
export PYTHONPATH=$(pwd)


conda activate scope

ulimit -S -n 131072
ulimit -S -u 1546461

srun python scope/train.py --c config/baobab_config.json