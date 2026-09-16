#!/bin/bash
#SBATCH --job-name=envcheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -e
cd ~/pytorch-lab
echo "node: $(hostname)"
nvidia-smi
source /hx2-weka/home/zf1225/miniforge3/etc/profile.d/conda.sh
conda activate torchlab
python 00_env/env_check.py
