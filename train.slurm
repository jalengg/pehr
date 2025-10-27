#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=promptehr_semantic_fix_aux0001
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

# Print job information
echo "========================================"
echo "PromptEHR Training Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working Directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "========================================"

# Show GPU information
echo "GPU Information:"
nvidia-smi
echo "========================================"

# Create logs and checkpoints directories
mkdir -p logs checkpoints

# Activate virtual environment
source venv/bin/activate

# Print Python and PyTorch info
echo "Environment:"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo "========================================"

# Run training
echo "Starting PromptEHR training..."
python trainer.py

# Print completion info
EXIT_CODE=$?
echo "========================================"
echo "Training completed"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================"
