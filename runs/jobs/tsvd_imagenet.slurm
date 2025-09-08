#!/bin/bash
#SBATCH -J tsvd_img
#SBATCH -t 48:00:00
#SBATCH -p swarm_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 task for 1 GPU
#SBATCH --gpus=1                  # 1 GPU total
#SBATCH --cpus-per-task=12
#SBATCH -o logs/tsvd_img.out   # Standard output file
#SBATCH -e logs/tsvd_img.err   # Standard error file

python train.py runs/configs/resnet/imagenet/resnet18_imagenet_tsvdlinear.yaml