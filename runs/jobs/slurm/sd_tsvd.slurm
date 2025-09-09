#!/bin/bash
#SBATCH -J sd_tsvd
#SBATCH -t 48:00:00
#SBATCH -p quad_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 task for 1 GPU
#SBATCH --gpus=1                  # 1 GPU total
#SBATCH --cpus-per-task=12
#SBATCH -o logs/sd_tsvd.out   # Standard output file
#SBATCH -e logs/sd_tsvd.err   # Standard error file

python train.py runs/configs/stablediffusion/imagenet/tsvd.yaml