# EHR Generation Training - Usage Guide

## Quick Start

### Single Training Run
```bash
# Submit a single job with default parameters
./single_job.sh

# Submit with custom parameters (num_patients, batch_size, epochs, lr, temp, top_k)
./single_job.sh 5000 32 50 0.0002 1.5 100
```

### Parameter Sweep
```bash
# Submit multiple jobs in parallel with different hyperparameters
./submit_param_sweep.sh
```

This will submit ~28 jobs testing different combinations of:
- Learning rates: [0.00005, 0.0001, 0.0002]
- Generation temperatures: [1.0, 1.3, 1.5]
- Top-k values: [30, 50, 100]
- Batch sizes: [16, 32]

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/train_<job_id>.out

# Check errors
tail -f logs/train_<job_id>.err

# View detailed training log
tail -f logs/train_<job_id>.log
```

## Diagnostic Logging

The training script includes extensive diagnostic logging to identify generation issues:

### Training Diagnostics
- Batch shape verification
- Special token counts
- Loss tracking per epoch
- Initial and final loss values

### Generation Diagnostics
- Token ID mappings for special tokens
- Top-20 next token predictions after prompt
- Probability distributions for critical tokens (<v>, <END>, PAD, EOS)
- Generated token counts and sequence analysis
- Warning if generation doesn't extend beyond prompt

### Log Levels
- **INFO**: High-level progress (console + file)
- **DEBUG**: Detailed diagnostics (file only)

## Key Issues to Watch For

Based on the generation failure after `<demo>` token, check logs for:

1. **Special token probabilities**: Are <v> and ICD codes getting learned?
   - Look for: "After prompt, probability of <v>: X.XXXXXX"
   - Should be high probability for <v> after <demo>

2. **Token embeddings**: Were special tokens properly initialized?
   - Look for: "Resized model embeddings to XXXXX tokens"
   - Verify token IDs are assigned

3. **Label masking**: Are padding tokens masked correctly?
   - Look for: "Special tokens in first batch"
   - masked_labels count should match padding

4. **Generation length**: Is model generating beyond forced tokens?
   - Look for: "Generation did not extend beyond forced initial tokens!"
   - This warning indicates the core issue

5. **Loss convergence**: Is training loss decreasing?
   - Check epoch-by-epoch loss in INFO logs
   - Plateaued loss might indicate learning issues

## Parameter Recommendations

Based on diagnostic findings, try:

### If model generates only <END> immediately:
- Increase warmup steps: `--num_warmup_steps 500`
- Lower temperature: `--generation_temp 0.8`
- Higher top_k: `--top_k 100`

### If training loss is high:
- More epochs: `--num_epochs 50`
- Lower learning rate: `--learning_rate 0.00005`
- Smaller batch size: `--batch_size 8`

### If training loss is low but generation fails:
- This suggests overfitting or special token embedding issues
- Try reducing epochs: `--num_epochs 20`
- Increase dataset size: `--num_patients 5000`

## Direct Python Execution

```bash
# Run without Slurm (for testing)
python main.py --num_epochs 5 --num_patients 100 --log_file test.log
```

## Output Files

- `logs/train_<job_id>.out`: Slurm stdout
- `logs/train_<job_id>.err`: Slurm stderr
- `logs/train_<job_id>.log`: Detailed training log with DEBUG info
- `training_<timestamp>.log`: Default log name if run without Slurm
