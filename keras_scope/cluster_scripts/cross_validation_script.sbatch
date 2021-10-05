#!/bin/bash
#SBATCH --job-name=cv_scope    # Job name
#SBATCH --partition=public-gpu
#SBATCH --gres=gpu:rtx:1                    # Number of GPUs
#SBATCH --ntasks=1
#SBATCH --mem=30G                     # Job memory request
#SBATCH --cpus-per-task=16
#SBATCH --time=47:59:00               # Time limit hrs:min:sec
#SBATCH --output=/home/users/k/klug/logs/scope/cv_keras_scope_%j.log   # Standard output and error log


module load Anaconda3

source /home/users/k/klug/.bashrc

cd /home/users/k/klug/scope
export PYTHONPATH=$(pwd)

conda activate scope

ulimit -S -n 131072
ulimit -S -u 1546461

srun python keras_scope/cross_validate.py -c keras_scope/configs/yggdrasil.json