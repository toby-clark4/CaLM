#!/bin/bash

#SBATCH --job-name=pl_CaLM
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH --time=25-00:00:00
#SBATCH --partition=rocm
#SBATCH --output=train_out.txt
#SBATCH --error=train_err.txt


# activate conda env
source ~/.bashrc
conda activate CaLM


srun python training.py
