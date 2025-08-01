#!/bin/bash
#SBATCH -J ditxl2_qat_ddp
#SBATCH -t 48:00:00
#SBATCH -p a100
#SBATCH --nodes 1                   # nodes
#SBATCH --ntasks-per-node=2         # 2 tasks for 2 GPUs
#SBATCH --gpus-per-task=1           # 1 GPU per task
#SBATCH --cpus-per-task=12
#SBATCH -o logs/ditxl2_qat_ddp.out   # Standard output file
#SBATCH -e logs/ditxl2_qat_ddp.err    # Standard error file