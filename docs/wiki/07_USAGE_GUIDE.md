# Usage Guide

**Last Updated:** October 24, 2025

This document provides step-by-step instructions for common tasks.

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installations
python --version  # Should be Python 3.12
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### 2. Verify Data

```bash
# Check MIMIC-III data files exist
ls -lh data_files/PATIENTS.csv.gz
ls -lh data_files/ADMISSIONS.csv.gz
ls -lh data_files/DIAGNOSES_ICD.csv.gz
```

## Training a Model

### Option 1: Train on Slurm (Recommended)

```bash
# 1. Modify config.py if needed
vim config.py  # Adjust hyperparameters

# 2. Submit training job
sbatch train.slurm

# 3. Monitor job
squeue -u $USER
tail -f logs/train_JOBID.out

# 4. Check for completion
# Model saved to /scratch/$USER/promptehr_checkpoints/best_model.pt
```

### Option 2: Train Locally (Small Scale)

```bash
# Reduce dataset size in config.py
# num_patients: 25000 → 1000

python trainer.py
```

**Expected Training Time:**
- 25k patients, 30 epochs: ~12-18 hours (GPU)
- 1k patients, 10 epochs: ~20 minutes (GPU)

## Generating Synthetic Patients

### Conditional Generation (Reconstruction)

**Purpose:** Reconstruct test patients from partial prompts

```python
from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_sequence_conditional
import torch

# 1. Load config and data
config = Config.from_defaults()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patient_records, vocab = load_mimic_data(
    patients_path=config.data.patients_path,
    admissions_path=config.data.admissions_path,
    diagnoses_path=config.data.diagnoses_path,
    num_patients=config.data.num_patients
)

# 2. Create tokenizer
tokenizer = DiagnosisCodeTokenizer(vocab)

# 3. Load model
checkpoint_path = "/scratch/$USER/promptehr_checkpoints/best_model.pt"
model = load_trained_model(checkpoint_path, tokenizer, config, device)

# 4. Generate
test_patient = patient_records[0]  # First patient
result = generate_patient_sequence_conditional(
    model=model,
    tokenizer=tokenizer,
    target_patient=test_patient,
    device=device,
    temperature=0.3,
    top_k=40,
    top_p=0.9,
    prompt_prob=0.5  # 50% codes as prompts
)

print(f"Generated {result['num_visits']} visits with {result['num_codes']} codes")
print(f"Target: {result['target_visits']}")
print(f"Generated: {result['generated_visits']}")
```

### Zero-Prompt Generation (Fully Synthetic)

**Purpose:** Generate patients from demographics only

```python
from generate import generate_patient_from_demographics

# Generate 10 synthetic patients
for i in range(10):
    result = generate_patient_from_demographics(
        model=model,
        tokenizer=tokenizer,
        device=device,
        age=None,  # Sample random age
        sex=None,  # Sample random sex
        temperature=0.7,
        top_k=40,
        top_p=0.9
    )

    print(f"\nPatient {i+1}:")
    print(f"  Demographics: {result['demographics']}")
    print(f"  Visits: {result['num_visits']}")
    print(f"  Total codes: {result['num_codes']}")
    for visit_idx, visit in enumerate(result['generated_visits']):
        print(f"  Visit {visit_idx+1}: {', '.join(visit[:5])}...")
```

### Quick Test Script

**File:** `test_unconditional.py`

```bash
python test_unconditional.py
```

Generates 10 synthetic patients and prints human-readable output.

## Evaluating Quality

### Medical Validity

**Evaluate age/sex appropriateness and duplicates:**

```bash
python evaluate_medical_validity.py
```

**Expected output:**
```
Medical Validity Evaluation
================================================================================
...
Pass/Fail Summary:
  ✓ Duplicate suppression
  ✓ Age appropriateness
  ✓ Sex appropriateness
  ✓ Jaccard similarity
```

### Semantic Coherence

**Evaluate code distributions and co-occurrence:**

```bash
python evaluate_semantic_coherence.py
```

**Expected output:**
```
Semantic Coherence Evaluation
================================================================================
...
SUMMARY
  Code Frequency Divergence: 0.30 (Good)
  Top-100 Overlap: 0.52 (Good)
  Co-occurrence Score: 25.3 (Good)
```

## Modifying Hyperparameters

### Adjust Training Configuration

**File:** `config.py`

**Common modifications:**

```python
@dataclass
class TrainingConfig:
    # Training duration
    num_epochs: int = 30  # 10-50

    # Model size
    batch_size: int = 8  # 4-16 (depends on GPU memory)

    # Learning
    learning_rate: float = 1e-4  # 5e-5 to 2e-4

    # Auxiliary losses (medical validity vs semantic coherence trade-off)
    age_loss_weight: float = 0.001  # 0.0001 to 0.01
    sex_loss_weight: float = 0.001  # 0.0001 to 0.01

    # Data corruption
    corruption_prob: float = 0.5  # 0.3-0.7
    lambda_poisson: float = 3.0   # 2.0-5.0
```

**After changing config:**

```bash
# Retrain model
sbatch train.slurm
```

### Adjust Generation Parameters

**In generation scripts or your code:**

```python
result = generate_patient_sequence_conditional(
    ...
    temperature=0.3,  # 0.1-2.0 (lower = more conservative)
    top_k=40,         # 10-100
    top_p=0.9,        # 0.7-1.0
    prompt_prob=0.5   # 0.0-1.0 (fraction of codes to reveal)
)
```

**Effects:**
- **Lower temperature (0.1-0.3):** More realistic, less diverse
- **Higher temperature (0.7-1.5):** More diverse, potentially less realistic
- **Lower prompt_prob (0.0):** Fully synthetic (zero-prompt)
- **Higher prompt_prob (0.8):** More faithful reconstruction

## Common Workflows

### Workflow 1: Train and Evaluate

```bash
# 1. Train model
sbatch train.slurm

# 2. Wait for completion (~12-18 hours)
tail -f logs/train_JOBID.out

# 3. Evaluate medical validity
python evaluate_medical_validity.py

# 4. Evaluate semantic coherence
python evaluate_semantic_coherence.py

# 5. If poor, adjust config and retrain
vim config.py  # Modify hyperparameters
sbatch train.slurm
```

### Workflow 2: Generate Synthetic Dataset

```bash
# 1. Load model and generate 1000 patients
python -c "
from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_from_demographics
import torch
import pickle

config = Config.from_defaults()
device = torch.device('cuda')

# Load
patient_records, vocab = load_mimic_data(...)
tokenizer = DiagnosisCodeTokenizer(vocab)
model = load_trained_model('checkpoints/best_model.pt', tokenizer, config, device)

# Generate
synthetic_patients = []
for i in range(1000):
    result = generate_patient_from_demographics(model, tokenizer, device)
    synthetic_patients.append(result)

# Save
with open('synthetic_patients.pkl', 'wb') as f:
    pickle.dump(synthetic_patients, f)
"

# 2. Use synthetic dataset
python your_analysis_script.py --data synthetic_patients.pkl
```

### Workflow 3: Hyperparameter Tuning

```bash
# 1. Create sweep of configurations
for aux_weight in 0.0001 0.001 0.01; do
    # Modify config
    sed -i "s/age_loss_weight: .*/age_loss_weight: $aux_weight/" config.py
    sed -i "s/sex_loss_weight: .*/sex_loss_weight: $aux_weight/" config.py

    # Train
    sbatch train.slurm

    # Rename checkpoint
    sleep 5  # Wait for job to start
    JOBID=$(squeue -u $USER --format="%i" | tail -1)
    echo "Training with aux_weight=$aux_weight, JOBID=$JOBID"
done

# 2. After all jobs complete, evaluate each
for checkpoint in checkpoints/best_model_*.pt; do
    python evaluate_medical_validity.py --checkpoint $checkpoint
    python evaluate_semantic_coherence.py --checkpoint $checkpoint
done
```

## Troubleshooting

### Issue: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# In config.py, reduce batch size
batch_size: int = 8  →  batch_size: int = 4
```

### Issue: Loss Diverges

**Symptoms:** Loss becomes NaN or increases

**Solutions:**
```python
# Reduce learning rate
learning_rate: float = 1e-4  →  learning_rate: float = 5e-5

# Check auxiliary loss weights (ensure not too high)
age_loss_weight: float = 0.001  # Should be << 1.0
sex_loss_weight: float = 0.001  # Should be << 1.0
```

### Issue: Poor Medical Validity

**Symptoms:** >5% age/sex violations

**Solutions:**
```python
# Increase auxiliary loss weights
age_loss_weight: float = 0.001  →  age_loss_weight: float = 0.005
sex_loss_weight: float = 0.001  →  sex_loss_weight: float = 0.005
```

### Issue: Poor Semantic Coherence

**Symptoms:** JS divergence >0.5, top-100 overlap <0.3

**Solutions:**
```python
# Reduce auxiliary loss weights (prioritize LM loss)
age_loss_weight: float = 0.01  →  age_loss_weight: float = 0.001
sex_loss_weight: float = 0.2   →  sex_loss_weight: float = 0.001

# Train longer
num_epochs: int = 30  →  num_epochs: int = 50
```

### Issue: Checkpoint Not Found

**Error:** `FileNotFoundError: checkpoints/best_model.pt`

**Solutions:**
```bash
# Check checkpoint directory
ls -lh /scratch/$USER/promptehr_checkpoints/

# Update path in config.py or script
checkpoint_dir: str = "/scratch/$USER/promptehr_checkpoints"
```

## Performance Benchmarks

**Training (25k patients, 30 epochs, NVIDIA A100):**
- Time: ~12-18 hours
- GPU memory: ~8GB
- Throughput: ~0.5 batch/sec

**Generation (100 patients):**
- Conditional (prompt_prob=0.5): ~2 minutes
- Zero-prompt: ~3 minutes

**Evaluation:**
- Medical validity (100 patients): ~1 minute
- Semantic coherence (100 patients): ~2 minutes

## Next Steps

- **Understand evaluation metrics:** See [Evaluation](06_EVALUATION.md)
- **Learn about generation modes:** See [Generation](05_GENERATION.md)
- **Dive into model details:** See [Model Architecture](03_MODEL_ARCHITECTURE.md)
- **Check deprecated features:** See [Deprecated History](08_DEPRECATED_HISTORY.md)
