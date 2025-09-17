#!/bin/bash

# Usage: ./submit_python.sh script.py [args...]
# Example: ./submit_python.sh train.py --epochs 100 --lr 0.001

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [arguments...]"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # Remove the script name from arguments
ARGS="$@"

# Set your default resources here (based on your typical setup)
NODES=1
NTASKS=1
GPUS=1
CPUS=12
TIME="24:00:00"
PARTITION="a100"

# Generate job name from script name
JOB_NAME=$(basename "$PYTHON_SCRIPT" .py)

# Ensure logs directory exists
mkdir -p logs

# Submit the job
sbatch \
    --job-name="$JOB_NAME" \
    --nodes="$NODES" \
    --ntasks="$NTASKS" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS" \
    --time="$TIME" \
    --partition="$PARTITION" \
    --output="logs/${JOB_NAME}.out" \
    --error="logs/${JOB_NAME}.err" \
    --wrap="python $PYTHON_SCRIPT $ARGS"

echo "Submitted job: $JOB_NAME"
echo "Resources: $GPUS GPU(s), $CPUS CPUs, $TIME time limit, partition: $PARTITION"
echo "Command: python $PYTHON_SCRIPT $ARGS"
echo "Logs: logs/${JOB_NAME}.out, logs/${JOB_NAME}.err"