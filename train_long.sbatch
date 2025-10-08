#!/bin/bash
#SBATCH --job-name=Tresette
#SBATCH --output=output.log
#SBATCH --error=errors.log
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --nice=0

cd /scratch.hpc/raffaele.neri2/Tresette_AI
source /scratch.hpc/raffaele.neri2/venv/bin/activate
python train_dqn.py