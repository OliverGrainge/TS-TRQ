#!/bin/bash
#SBATCH -J tsvd_vit
#SBATCH -t 48:00:00
#SBATCH -p a100
#SBATCH --nodes 1                   # nodes
#SBATCH --ntasks-per-node=1         # only 1 task for single GPU
#SBATCH --gpus-per-task=1           # 1 GPU per task
#SBATCH --cpus-per-task=12


# SLURM will launch 1 task with 1 GPU.
python train.py configs/train_vit_cifar10_tsvdlinear_reg1.yaml
python train.py configs/train_vit_cifar10_tsvdlinear_reg2.yaml
python train.py configs/train_vit_cifar10_tsvdlinear_reg3.yaml
python train.py configs/train_vit_cifar10_tsvdlinear_reg4.yaml
python train.py configs/train_vit_cifar10_tsvdlinear_reg5.yaml

