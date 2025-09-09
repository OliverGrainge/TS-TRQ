#!/bin/bash
#SBATCH -J sdxl
#SBATCH -t 48:00:00
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH -o logs/sdxl.out
#SBATCH -e logs/sdxl.err

python scratch.py