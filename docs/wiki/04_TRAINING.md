# Training

**Last Updated:** October 24, 2025

This document describes the training process, optimization strategy, and metrics tracking.

## Training Pipeline

```
Load Data → Create Dataset → Create DataLoader → Initialize Model → Train Loop → Validate → Save Checkpoint
```

## 1. Training Configuration

**File:** `config.py`

```python
@dataclass
class TrainingConfig:
    # Optimization
    batch_size: int = 8              # Effective 24-32 with sample expansion
    num_epochs: int = 30             # Full training to convergence
    learning_rate: float = 1e-4      # AdamW learning rate
    weight_decay: float = 0.01       # L2 regularization
    warmup_steps: int = 1000         # Linear warmup
    max_grad_norm: float = 1.0       # Gradient clipping

    # Checkpointing
    checkpoint_dir: str = "/scratch/jalenj4/promptehr_checkpoints"
    save_every_n_epochs: int = 10

    # Logging
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 1

    # Reconstruction Jaccard
    compute_reconstruction_jaccard: bool = True
```

## 2. Training Loop (trainer.py)

### Initialization

```python
# 1. Load data
patient_records, vocab = load_mimic_data(...)

# 2. Create tokenizer
tokenizer = DiagnosisCodeTokenizer(vocab)

# 3. Create dataset
dataset = EHRPatientDataset(patient_records)

# 4. Split train/validation (80/20)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 5. Create data collator with corruptions
collator = EHRDataCollator(
    tokenizer=tokenizer,
    lambda_poisson=3.0,
    mask_probability=0.15,
    del_probability=0.15,
    rep_probability=0.15,
    corruption_prob=0.5
)

# 6. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collator)

# 7. Initialize model
model = PromptBartWithDemographicPrediction(
    tokenizer=tokenizer,
    config=config
).to(device)

# 8. Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 9. Learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

### Training Epoch

```python
for epoch in range(num_epochs):
    model.train()
    train_metrics = MetricsTracker()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        # 1. Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        x_num = batch['x_num'].to(device)
        x_cat = batch['x_cat'].to(device)

        # 2. Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat
        )

        # 3. Backward pass
        loss = outputs['loss']
        loss.backward()

        # 4. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 6. Track metrics
        train_metrics.update(
            loss=loss.item(),
            lm_loss=outputs['lm_loss'],
            age_loss=outputs['age_loss'],
            sex_loss=outputs['sex_loss'],
            learning_rate=scheduler.get_last_lr()[0]
        )

        # 7. Log progress
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch+1}, Step {batch_idx+1}/{len(train_loader)} - "
                f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Validation
    val_metrics = validate(model, val_loader, device, logger, config)

    # Save checkpoint
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        save_checkpoint(model, optimizer, epoch, val_metrics, 'best_model.pt')
```

### Validation Loop

```python
def validate(model, val_loader, device, logger, config):
    model.eval()
    val_metrics = MetricsTracker()

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass (same as training)
            outputs = model(...)

            # Track metrics
            val_metrics.update(
                loss=outputs['loss'].item(),
                lm_loss=outputs['lm_loss'],
                age_loss=outputs['age_loss'],
                sex_loss=outputs['sex_loss']
            )

    # Compute perplexity
    perplexity = compute_perplexity(val_metrics.lm_losses)

    # Compute temporal perplexity (if enabled)
    if config.training.compute_tpl:
        tpl = compute_temporal_perplexity(model, val_loader, device)

    # Compute reconstruction Jaccard (if enabled)
    if config.training.compute_reconstruction_jaccard:
        jaccard = compute_reconstruction_jaccard(
            model, val_patient_records, tokenizer, device, logger
        )

    return {
        'loss': val_metrics.get_average('loss'),
        'lm_loss': val_metrics.get_average('lm_loss'),
        'age_loss': val_metrics.get_average('age_loss'),
        'sex_loss': val_metrics.get_average('sex_loss'),
        'perplexity': perplexity,
        'tpl': tpl,
        'reconstruction_jaccard': jaccard
    }
```

## 3. Optimization Strategy

### AdamW Optimizer

**Choice:** AdamW (Adam with decoupled weight decay)

**Parameters:**
- Learning rate: 1e-4
- Weight decay: 0.01 (L2 regularization)
- Betas: (0.9, 0.999) - default Adam momentum
- Epsilon: 1e-8 - numerical stability

**Why AdamW?**
- Better generalization than Adam (decoupled weight decay)
- Standard choice for transformer models
- Stable training with large models

### Learning Rate Schedule

**Strategy:** Linear warmup + linear decay

```python
# Warmup: 0 → 1e-4 over 1000 steps
# Decay: 1e-4 → 0 over remaining steps

lr_t = (1e-4) * min(step / 1000, 1 - (step - 1000) / (total_steps - 1000))
```

**Why Warmup?**
- Prevents early instability (large gradients at initialization)
- Standard practice for transformer training
- 1000 steps ≈ 0.4 epochs (sufficient for stabilization)

### Gradient Clipping

**max_norm:** 1.0

**Why?**
- Prevents gradient explosion (especially with auxiliary losses)
- Standard practice for RNN/transformer training
- Empirically found 1.0 to be stable

## 4. Metrics Tracking

### MetricsTracker Class (metrics.py)

**Tracked Metrics:**

```python
class MetricsTracker:
    def __init__(self):
        self.losses = []
        self.lm_losses = []
        self.age_losses = []
        self.sex_losses = []
        self.perplexities = []
        self.reconstruction_jaccards = []

    def update(self, loss, lm_loss, age_loss, sex_loss, **kwargs):
        self.losses.append(loss)
        self.lm_losses.append(lm_loss)
        self.age_losses.append(age_loss)
        self.sex_losses.append(sex_loss)

    def get_average(self, metric_name):
        return np.mean(getattr(self, f"{metric_name}s"))
```

### Perplexity

**Language Model Perplexity:**

```python
def compute_perplexity(lm_losses):
    avg_loss = np.mean(lm_losses)
    return np.exp(avg_loss)
```

**Interpretation:**
- Lower is better
- Measures model's uncertainty in next-token prediction
- Target: <10 (well-calibrated), <5 (excellent)

### Temporal Perplexity (TPL)

**Purpose:** Measure next-visit prediction quality

```python
def compute_temporal_perplexity(model, val_loader, device):
    # Use only next-visit prediction samples
    tpl_losses = []

    for batch in val_loader:
        if batch['task_type'] == 'next_visit_prediction':
            outputs = model(...)
            tpl_losses.append(outputs['lm_loss'])

    return np.exp(np.mean(tpl_losses))
```

**Interpretation:**
- Measures temporal coherence (can model predict next visit?)
- Target: Similar to LM perplexity (~1.0-1.5)

### Reconstruction Jaccard

**Purpose:** Measure prompt-aware reconstruction quality

```python
def compute_reconstruction_jaccard(model, patient_records, tokenizer, device, logger):
    jaccards = []

    for patient in random.sample(patient_records, 100):
        # Generate with prompt_prob=0.5 (50% of codes as prompts)
        result = generate_patient_sequence_conditional(
            model, tokenizer, patient, device, prompt_prob=0.5
        )

        # Compute Jaccard(generated - prompts, target - prompts)
        for visit_idx, (gen_visit, tgt_visit, prompt_codes) in enumerate(...):
            gen_new = set(gen_visit) - set(prompt_codes)
            tgt_new = set(tgt_visit) - set(prompt_codes)

            if len(gen_new) > 0 or len(tgt_new) > 0:
                jaccard = len(gen_new & tgt_new) / len(gen_new | tgt_new)
                jaccards.append(jaccard)

    return np.mean(jaccards)
```

**Interpretation:**
- Measures reconstruction quality excluding prompt codes
- Target: 0.40-0.45 (good reconstruction)
- Unlike standard Jaccard, this measures true generation capability

## 5. Checkpointing

### Checkpoint Structure

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': val_loss,
    'config': config
}
torch.save(checkpoint, checkpoint_path)
```

### Checkpoint Strategy

1. **best_model.pt**: Lowest validation loss
2. **checkpoint_epoch_N.pt**: Every 10 epochs (backup)
3. **Location:** `/scratch/jalenj4/promptehr_checkpoints/`

### Loading Checkpoint

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 6. Training on Slurm

**Script:** `train.slurm`

```bash
#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=promptehr_semantic_fix_aux0001
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com

source venv/bin/activate
python trainer.py
```

**Submit:**
```bash
sbatch train.slurm
```

**Monitor:**
```bash
squeue -u jalenj4
tail -f logs/train_JOBID.out
```

## 7. Training Timeline

**Typical Training Run (25k patients, 30 epochs):**

- **Setup:** 1-2 minutes (data loading, model initialization)
- **Per Epoch:** ~20-30 minutes (2500 batches × 0.5s/batch)
- **Validation:** ~2 minutes per epoch
- **Total:** ~12-18 hours for 30 epochs

**GPU Utilization:**
- Memory: ~8GB / 32GB (NVIDIA A100)
- Utilization: ~80-90% (compute-bound)

## 8. Training Curves

**Expected Convergence:**

```
Epoch 1:  Loss 7.85 → LM 7.85, Age 1.79, Sex 0.70
Epoch 5:  Loss 1.20 → LM 1.20, Age 0.85, Sex 0.15
Epoch 10: Loss 0.05 → LM 0.05, Age 0.30, Sex 0.05
Epoch 20: Loss 0.01 → LM 0.01, Age 0.15, Sex 0.02
Epoch 30: Loss 0.005 → LM 0.005, Age 0.10, Sex 0.01
```

**Loss plateaus around epoch 30** - further training provides minimal improvement.

## 9. Common Issues

### Issue 1: Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
- Reduce batch size (8 → 4)
- Reduce max_length (512 → 256)
- Enable gradient checkpointing (slower but less memory)

### Issue 2: Loss Divergence

**Symptoms:** Loss increases or becomes NaN

**Solutions:**
- Check learning rate (reduce if too high)
- Check gradient clipping (ensure max_norm=1.0)
- Check auxiliary loss weights (ensure not too high)

### Issue 3: Slow Training

**Symptoms:** <0.5 batch/sec

**Solutions:**
- Check GPU utilization (`nvidia-smi`)
- Reduce number of corruptions (faster collation)
- Increase num_workers in DataLoader (CPU parallelism)

## 10. Hyperparameter Tuning

**Critical Hyperparameters:**

| Parameter | Default | Tuning Range | Impact |
|-----------|---------|--------------|--------|
| `learning_rate` | 1e-4 | 5e-5 to 2e-4 | Too high: divergence, Too low: slow convergence |
| `age_loss_weight` | 0.001 | 0.0001 to 0.01 | Higher: more medical validity, less semantic coherence |
| `sex_loss_weight` | 0.001 | 0.0001 to 0.01 | Higher: more medical validity, less semantic coherence |
| `corruption_prob` | 0.5 | 0.3 to 0.7 | Higher: more diversity, slower training |
| `lambda_poisson` | 3.0 | 2.0 to 5.0 | Higher: larger masked spans, harder task |

**Tuning Strategy:**
1. Fix architecture, tune LR first
2. Tune auxiliary loss weights for validity/coherence trade-off
3. Fine-tune corruption parameters last

## Next Steps

- **Learn about generation:** See [Generation](05_GENERATION.md)
- **Understand evaluation:** See [Evaluation](06_EVALUATION.md)
- **See model details:** See [Model Architecture](03_MODEL_ARCHITECTURE.md)
