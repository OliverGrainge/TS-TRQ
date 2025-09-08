#!/bin/bash
#SBATCH -J t_img
#SBATCH -t 24:00:00
#SBATCH -p a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1                # 1 task for 1 GPU
#SBATCH --gpus=1                  # 1 GPU total
#SBATCH --cpus-per-task=12
#SBATCH -o logs/t_img.out   # Standard output file
#SBATCH -e logs/t_img.err   # Standard error file


python train.py runs/configs/resnet/imagenet/resnet18_imagenet_tlinear.yaml