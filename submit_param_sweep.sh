#!/bin/bash
# Revised parameter sweep based on diagnostic findings

mkdir -p logs

echo "Starting revised parameter sweep at $(date)"
echo "Submitting jobs to Slurm..."

NUM_PATIENTS=3000
BATCH_SIZE=16

# Revised grid: epochs and learning rate
EPOCHS=(10 15 20)
LEARNING_RATES=(0.00005 0.0001)
GENERATION_TEMP=1.0
TOP_K=50

job_count=0

# Epoch + LR sweep
for epochs in "${EPOCHS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        job_name="ehr_e${epochs}_lr${lr}"

        echo "Submitting job: $job_name"
        echo "  EPOCHS=$epochs, LR=$lr"

        sbatch --job-name="$job_name" \
               run_training.slurm \
               $NUM_PATIENTS \
               $BATCH_SIZE \
               $epochs \
               $lr \
               $GENERATION_TEMP \
               $TOP_K

        job_count=$((job_count + 1))
        sleep 0.5
    done
done

echo ""
echo "Revised sweep submission complete!"
echo "Total jobs submitted: $job_count"
echo "End time: $(date)"
echo ""
echo "Check job status with: squeue -u $USER"
echo "Monitor logs in: logs/"
