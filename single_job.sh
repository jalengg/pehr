#!/bin/bash
# Quick script to submit a single training job with custom parameters
# Usage: ./single_job.sh [num_patients] [batch_size] [num_epochs] [learning_rate] [temp] [top_k]

# Create logs directory
mkdir -p logs

# Parse arguments with defaults
NUM_PATIENTS=${1:-3000}
BATCH_SIZE=${2:-16}
NUM_EPOCHS=${3:-30}
LEARNING_RATE=${4:-0.0001}
GENERATION_TEMP=${5:-1.3}
TOP_K=${6:-50}

JOB_NAME="ehr_single_$(date +%Y%m%d_%H%M%S)"

echo "Submitting single training job: $JOB_NAME"
echo "Parameters:"
echo "  NUM_PATIENTS: $NUM_PATIENTS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  GENERATION_TEMP: $GENERATION_TEMP"
echo "  TOP_K: $TOP_K"
echo ""

sbatch --job-name="$JOB_NAME" \
       run_training.slurm \
       $NUM_PATIENTS \
       $BATCH_SIZE \
       $NUM_EPOCHS \
       $LEARNING_RATE \
       $GENERATION_TEMP \
       $TOP_K

echo "Job submitted. Check status with: squeue -u $USER"
