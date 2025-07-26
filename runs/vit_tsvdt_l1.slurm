#!/bin/bash
#SBATCH -J tsvdt_l1
#SBATCH -t 48:00:00
#SBATCH -p swarm_a100
#SBATCH --nodes 1                   # nodes
#SBATCH --ntasks-per-node=1         # only 1 task for single GPU
#SBATCH --gpus-per-task=1           # 1 GPU per task
#SBATCH --cpus-per-task=12


# SLURM will launch 1 task with 1 GPU.
python train.py configs/vit_cifar100_l1_tsvdt/reg1.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg2.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg3.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg4.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg5.yaml


