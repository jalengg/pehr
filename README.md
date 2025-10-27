# PromptEHR: Synthetic EHR Generation with Demographic Conditioning

A BART-based system for generating synthetic Electronic Health Records (EHR) with demographic conditioning using MIMIC-III clinical data.

## Overview

PromptEHR generates realistic patient sequences containing:
- Demographics (age, sex)
- Medical visit histories
- ICD-9 diagnosis codes (5,562 vocabulary)

**Key Features:**
- No code fragmentation (1:1 token mapping)
- Multi-task learning (language modeling + age/sex prediction)
- Dual prompt conditioning (encoder + decoder)
- Two generation modes: conditional reconstruction and zero-prompt synthesis

## Quick Start

### Training

```bash
# Submit training job to Slurm
sbatch train.slurm

# Monitor progress
tail -f logs/train_JOBID.out
```

### Generating Patients

```bash
# Generate 10 synthetic patients
python test_unconditional.py

# Evaluate medical validity (100 patients)
python evaluate_medical_validity.py

# Evaluate semantic coherence (100 patients)
python evaluate_semantic_coherence.py
```

## Current Performance

**Model:** PromptBartWithDemographicPrediction v4 (245M parameters)
**Training Data:** MIMIC-III (25,000 patients)

**Medical Validity:**
- Age-appropriate: 99%
- Sex-appropriate: 96%
- Duplicate codes: 0%
- Reconstruction Jaccard: 0.40-0.45

**Semantic Coherence:**
- Currently suboptimal (active training to improve)
- JS divergence: 0.61 (target: <0.3)
- Top-100 overlap: 0.04 (target: >0.5)

## Architecture

```
MIMIC-III Data
    ↓
Data Loader → Vocabulary → Tokenizer → Dataset
    ↓
PromptBartWithDemographicPrediction
  ├── ConditionalPrompt (reparameterization)
  ├── PromptBartEncoder (prompt injection)
  ├── PromptBartDecoder (prompt injection)
  └── Multi-task heads (LM + age + sex)
    ↓
Generation (Conditional or Zero-Prompt)
    ↓
Evaluation (Medical Validity + Semantic Coherence)
```

**Loss Function:**
```python
total_loss = lm_loss + 0.001 × age_loss + 0.001 × sex_loss
```

## Documentation

**Complete documentation in `docs/wiki/`:**

- [00_INDEX.md](docs/wiki/00_INDEX.md) - Documentation overview
- [01_ARCHITECTURE.md](docs/wiki/01_ARCHITECTURE.md) - System design
- [02_DATA_PIPELINE.md](docs/wiki/02_DATA_PIPELINE.md) - Data processing
- [03_MODEL_ARCHITECTURE.md](docs/wiki/03_MODEL_ARCHITECTURE.md) - Neural network details
- [04_TRAINING.md](docs/wiki/04_TRAINING.md) - Training process
- [05_GENERATION.md](docs/wiki/05_GENERATION.md) - Patient generation
- [06_EVALUATION.md](docs/wiki/06_EVALUATION.md) - Quality metrics
- [07_USAGE_GUIDE.md](docs/wiki/07_USAGE_GUIDE.md) - How to use the system
- [08_DEPRECATED_HISTORY.md](docs/wiki/08_DEPRECATED_HISTORY.md) - Evolution timeline

**Additional documentation:**
- `docs/reference/` - Medical validity rules, multi-task learning details
- `docs/historical/` - Implementation history (phases 1-9)
- `docs/analysis/` - Quality assessments and training results

## File Structure

```
pehr_scratch/
├── docs/
│   ├── wiki/           # Primary documentation
│   ├── reference/      # Technical references
│   ├── historical/     # Implementation history
│   └── analysis/       # Quality assessments
├── deprecated/
│   ├── legacy_implementations/  # Old text-based approach
│   ├── unit_tests/             # Phase 1-3 validation tests
│   ├── utilities/              # One-off scripts
│   └── backups/                # v2, v3 snapshots
├── data_files/         # MIMIC-III CSVs
├── data_splits/        # Train/test pickles
├── checkpoints/        # Local model checkpoints
├── logs/               # Training logs
├── trainer.py          # Training loop
├── generate.py         # Patient generation
├── evaluate_*.py       # Evaluation scripts
└── [other Python modules]
```

## Key Components

**Active Python Files:**

| File | Purpose |
|------|---------|
| `trainer.py` | Training loop, optimization, validation |
| `config.py` | Hyperparameters and configuration |
| `data_loader.py` | MIMIC-III data loading |
| `vocabulary.py` | Diagnosis code vocabulary (1:1 mapping) |
| `code_tokenizer.py` | Tokenizer without fragmentation |
| `dataset.py` | PyTorch dataset with corruptions |
| `prompt_bart_model.py` | Main model with multi-task learning |
| `prompt_bart_encoder.py` | BART encoder with prompt injection |
| `prompt_bart_decoder.py` | BART decoder with prompt injection |
| `conditional_prompt.py` | Demographic embeddings |
| `metrics.py` | Loss tracking, perplexity, Jaccard |
| `generate.py` | Conditional and zero-prompt generation |
| `evaluate_medical_validity.py` | Age/sex appropriateness |
| `evaluate_semantic_coherence.py` | Distribution matching |

## Generation Modes

### 1. Conditional Generation

Reconstruct patient from partial code prompts:

```python
from generate import generate_patient_sequence_conditional

result = generate_patient_sequence_conditional(
    model=model,
    tokenizer=tokenizer,
    target_patient=test_patient,
    device=device,
    prompt_prob=0.5  # Reveal 50% of codes
)
```

### 2. Zero-Prompt Generation

Generate from demographics only:

```python
from generate import generate_patient_from_demographics

result = generate_patient_from_demographics(
    model=model,
    tokenizer=tokenizer,
    device=device,
    age=65.0,
    sex=0  # Male
)
```

## Evaluation Metrics

**Medical Validity:**
- Age appropriateness (>98% target)
- Sex appropriateness (>99% target)
- Duplicate suppression (<1% target)
- Reconstruction Jaccard (0.40-0.45 target)

**Semantic Coherence:**
- JS divergence (<0.3 target)
- Distribution match (KS tests, p>0.05 target)
- Top-100 code overlap (>0.5 target)
- Co-occurrence score (>20 target)

## Technical Highlights

**1. No Code Fragmentation**
- Each ICD-9 code = single token ID
- Prevents "401.9" → ["401", ".", "9"] fragmentation
- Enables gradient-based learning of code embeddings

**2. Dual Prompt Conditioning**
- Separate demographic prompts for encoder and decoder
- Reduces demographic drift in long sequences
- Reparameterization with d_hidden=128 bottleneck

**3. Multi-Task Learning**
- Token-level age/sex prediction
- Low auxiliary weights (0.001) prioritize LM learning
- Balances medical validity with semantic coherence

**4. Duplicate Suppression**
- Uses `no_repeat_ngram_size=1` during generation
- Prevents duplicate codes within single visit

## Requirements

- Python 3.12
- PyTorch
- HuggingFace Transformers
- MIMIC-III access (physionet.org)
- GPU recommended (training: ~12-18 hours for 30 epochs)

## Current Limitations

1. **Semantic coherence needs improvement** - Generated codes medically valid but statistically implausible
2. **Architecture inefficiency** - BART encoder-decoder less efficient than GPT-style decoder-only
3. **Medical invalidity** - ~1-4% codes remain age/sex-inappropriate
4. **No medical ontology** - Lacks ICD-9 hierarchy, SNOMED/UMLS integration

See `docs/wiki/08_DEPRECATED_HISTORY.md` for lessons learned.

## Citation

Based on PromptEHR methodology. If you use this code, please acknowledge:
- MIMIC-III: Johnson et al. (2016)
- BART: Lewis et al. (2020)
- PromptEHR: Original prompt-based EHR generation approach

## License

This project uses MIMIC-III data, which requires PhysioNet credentialed access and adherence to data use agreements.

## Contact

For questions or issues, see documentation in `docs/wiki/` or check `CHANGELOG.md` for recent updates.

## Acknowledgments

- MIMIC-III dataset (MIT-LCP)
- HuggingFace Transformers library
- IllinoisComputes GPU cluster
