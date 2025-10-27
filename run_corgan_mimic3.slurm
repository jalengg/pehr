#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=corgan_mimic3
#SBATCH --output=logs/corgan_mimic3_%j.out
#SBATCH --error=logs/corgan_mimic3_%j.err
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /u/jalenj4/PyHealth
source pyhealth/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth/PyHealth:$PYTHONPATH

mkdir -p logs corgan_results

echo "Starting CorGAN synthetic EHR generation..."
python examples/synthetic_data_generation_mimic3_corgan.py \
    --data_path ./data_files \
    --output_path ./corgan_results \
    --ae_epochs 10 \
    --gan_epochs 50 \
    --n_samples 1000 \
    --batch_size 128 \
    --latent_dim 128 \
    --lr 0.0001

echo "CorGAN training completed!"
echo "Results saved to: ./corgan_results/"