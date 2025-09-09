#!/bin/bash
#SBATCH -J sd_tsvd_low
#SBATCH -t 48:00:00
#SBATCH -p swarm_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 task for 1 GPU
#SBATCH --gpus=1                  # 1 GPU total
#SBATCH --cpus-per-task=12
#SBATCH -o logs/sd_tsvd_low.out   # Standard output file
#SBATCH -e logs/sd_tsvd_low.err   # Standard error file

python train.py runs/configs/stablediffusion/imagenet/tsvd_low.yaml