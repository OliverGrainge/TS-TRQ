##!/bin/bash
#SBATCH -J dit_imagenet
#SBATCH -t 48:00:00
#SBATCH -N 1                        # nodes
#SBATCH --ntasks-per-node=4         # tasks = gpus per node
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# SLURM will launch 16 tasks (2*8). Each task runs python once.
srun python -u train.py \
  --batch_size 8 \
  --accelerator gpu \
  --devices 1 \
  --strategy ddp \
  --precision 32