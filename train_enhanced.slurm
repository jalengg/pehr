#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=promptehr_enhanced
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=logs/train_enhanced_%j.out
#SBATCH --error=logs/train_enhanced_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

echo "=================================================="
echo "PromptEHR Enhanced Training (Multi-Task + TPL)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working Directory: $(pwd)"
echo "Start time: $(date)"
echo ""

# Create directories
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi

# Print Python packages
echo ""
echo "Python version:"
python --version

echo ""
echo "Key packages:"
pip list | grep -E "torch|transformers|numpy|pandas"

echo ""
echo "=================================================="
echo "Starting Training"
echo "=================================================="
echo ""

# Run training with enhanced strategy
python trainer.py

exit_code=$?

echo ""
echo "=================================================="
echo "Training Complete"
echo "=================================================="
echo "Exit code: $exit_code"
echo "End time: $(date)"

# Print final GPU stats
echo ""
echo "Final GPU state:"
nvidia-smi

exit $exit_code
