#!/bin/bash
#SBATCH --job-name=hyperopt_scope    # Job name
#SBATCH --partition=public-gpu
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --ntasks=1
#SBATCH --mem=30G                     # Job memory request
#SBATCH --time=:2-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/users/k/klug/logs/scope/hyperopt_scope_%j.log   # Standard output and error log


module load Anaconda3

source /home/users/k/klug/.bashrc

cd /home/users/k/klug/scope
export PYTHONPATH=$(pwd)

conda activate scope

ulimit -S -n 131072
ulimit -S -u 1546461

srun python scope/hyperopt.py --c /home/users/k/klug/scope/scope/configs/hyperopt_config.json