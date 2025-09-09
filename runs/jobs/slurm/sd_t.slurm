#!/bin/bash
#SBATCH -J sd_t
#SBATCH -t 48:00:00
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 task for 1 GPU
#SBATCH --gpus=1                  # 1 GPU total
#SBATCH --cpus-per-task=12
#SBATCH -o logs/sd_t.out   # Standard output file
#SBATCH -e logs/sd_t.err   # Standard error file

python train.py runs/configs/stablediffusion/imagenet/t.yaml