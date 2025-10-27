#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=ehr_gen_train
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

cd "$SLURM_SUBMIT_DIR"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Show GPU info
nvidia-smi

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Parse command-line arguments with defaults
NUM_PATIENTS=${1:-3000}
BATCH_SIZE=${2:-16}
NUM_EPOCHS=${3:-30}
LEARNING_RATE=${4:-0.0001}
GENERATION_TEMP=${5:-1.3}
TOP_K=${6:-50}

echo "Parameters:"
echo "  NUM_PATIENTS: $NUM_PATIENTS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  GENERATION_TEMP: $GENERATION_TEMP"
echo "  TOP_K: $TOP_K"

# Run the training script
python main.py \
    --num_patients $NUM_PATIENTS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --generation_temp $GENERATION_TEMP \
    --top_k $TOP_K \
    --log_file "logs/train_${SLURM_JOB_ID}.log"

echo "End Time: $(date)"
echo "Exit Code: $?"
