#!/bin/bash
#SBATCH -J sd_fine_ddp
#SBATCH -t 48:00:00
#SBATCH -p a100
#SBATCH --nodes=2                 # Use 2 nodes for multi-node DDP
#SBATCH --ntasks-per-node=2       # 2 GPUs per node
#SBATCH --gpus-per-node=2         # 2 GPUs per node
#SBATCH --cpus-per-task=12        # CPUs per GPU task
#SBATCH -o logs/unet_fp_ddp.out   # Standard output file
#SBATCH -e logs/unet_fp_ddp.err   # Standard error file

# Run the training script
srun python train.py runs/configs/diffusion/fp_ddp.yaml