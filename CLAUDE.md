# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptEHR is a synthetic Electronic Health Record (EHR) generation system based on fine-tuned BART with demographic conditioning. It learns from MIMIC-III clinical data to generate realistic patient sequences containing demographics and medical visit histories with ICD-9 diagnosis codes.

**Current Model Version:** PromptBartWithDemographicPrediction v4
**Training Data:** MIMIC-III (25,000 patients)
**Vocabulary:** 5,562 unique ICD-9 diagnosis codes

## Quick Navigation

**Comprehensive documentation is available in `docs/wiki/`:**

- **[Index](docs/wiki/00_INDEX.md)** - Complete documentation overview
- **[Architecture](docs/wiki/01_ARCHITECTURE.md)** - System design and components
- **[Data Pipeline](docs/wiki/02_DATA_PIPELINE.md)** - Data processing
- **[Model Architecture](docs/wiki/03_MODEL_ARCHITECTURE.md)** - Neural network details
- **[Training](docs/wiki/04_TRAINING.md)** - Training process
- **[Generation](docs/wiki/05_GENERATION.md)** - Patient generation
- **[Evaluation](docs/wiki/06_EVALUATION.md)** - Quality metrics
- **[Usage Guide](docs/wiki/07_USAGE_GUIDE.md)** - How to use the system
- **[Deprecated History](docs/wiki/08_DEPRECATED_HISTORY.md)** - Evolution timeline

## Current Architecture (High-Level)

### Data Representation

**Modern approach (current):**
```python
PatientRecord(
    subject_id="12345",
    visits=[
        ["401.9", "250.00", "585.9"],  # Visit 1
        ["428.0", "584.9"]              # Visit 2
    ],
    age=65.0,
    gender=0  # 0=M, 1=F
)
```

**Each ICD-9 code = single token ID** (no fragmentation)

**Old text-based approach (deprecated):**
```
"65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 584.9 <\v> <END>"
```
Why deprecated: BART tokenizer fragmented codes ("401.9" → ["401", ".", "9"])

### Model Components

1. **Data Processing:**
   - `data_loader.py` - MIMIC-III data loading
   - `vocabulary.py` - Diagnosis code vocabulary (1:1 mapping)
   - `code_tokenizer.py` - Tokenizer without fragmentation
   - `dataset.py` - PyTorch dataset with data corruption

2. **Model Architecture:**
   - `prompt_bart_model.py` - Main model with multi-task learning
   - `prompt_bart_encoder.py` - BART encoder with prompt injection
   - `prompt_bart_decoder.py` - BART decoder with prompt injection
   - `conditional_prompt.py` - Demographic embedding (reparameterization)

3. **Training & Evaluation:**
   - `trainer.py` - Training loop, optimization
   - `config.py` - Hyperparameters
   - `metrics.py` - Loss tracking, perplexity, Jaccard
   - `evaluate_medical_validity.py` - Age/sex appropriateness
   - `evaluate_semantic_coherence.py` - Distribution matching

4. **Generation:**
   - `generate.py` - Conditional and zero-prompt generation

## Running the Code

### Training

```bash
# Submit to Slurm cluster
sbatch train.slurm

# Monitor
squeue -u $USER
tail -f logs/train_JOBID.out
```

**Configuration:** Edit `config.py` to adjust hyperparameters

### Generating Patients

```bash
# Quick test (10 synthetic patients)
python test_unconditional.py

# Medical validity evaluation (100 patients)
python evaluate_medical_validity.py

# Semantic coherence evaluation (100 patients)
python evaluate_semantic_coherence.py
```

### Generation Modes

1. **Conditional (Reconstruction):** Reconstruct patient from partial code prompts
2. **Zero-Prompt (Fully Synthetic):** Generate from demographics only (age, sex)

See `docs/wiki/05_GENERATION.md` for details.

## Key Implementation Details

### Special Tokens

Seven special tokens (0-6):
- `<s>` (BOS), `<pad>`, `</s>` (EOS), `<unk>`, `<v>`, `<\v>`, `<mask>`

Diagnosis codes start at token ID 7 (code_offset=7)

### Token Sequence Format

```
<s> <v> code1 code2 <\v> <v> code3 code4 <\v> </s>
```

### Multi-Task Loss

```python
total_loss = lm_loss + 0.001 × age_loss + 0.001 × sex_loss
```

**Why low weights (0.001)?**
- Prioritize learning realistic code distributions (LM loss)
- Auxiliary losses provide weak guidance for medical validity
- High weights (0.01-0.2) destroy semantic coherence

### Label Masking

**Critical:** Padding tokens must be set to -100 in labels
```python
labels[labels == tokenizer.pad_token_id] = -100
```

Prevents model from learning to predict padding.

### Duplicate Suppression

Generation uses `no_repeat_ngram_size=1` to prevent duplicate codes within a visit.

## File Organization

```
pehr_scratch/
├── docs/
│   ├── wiki/           # Comprehensive documentation (START HERE)
│   ├── reference/      # Medical validity rules, multi-task learning
│   ├── historical/     # Implementation history
│   └── analysis/       # Quality assessments
├── deprecated/
│   ├── legacy_implementations/  # Old main.py (text-based)
│   ├── unit_tests/             # Phase 1-3 tests
│   ├── utilities/              # One-off scripts
│   └── backups/                # v2, v3 directories
├── data_files/         # MIMIC-III CSVs
├── data_splits/        # Train/test split pickles
├── checkpoints/        # Local checkpoints
├── logs/               # Training logs
├── [active Python files]
├── CLAUDE.md           # This file
├── CHANGELOG.md        # Change history
└── README.md           # Project overview
```

## Current Performance

**Medical Validity (Latest):**
- Age-appropriate: 99%
- Sex-appropriate: 96%
- Duplicate rate: 0%
- Jaccard similarity: 0.40-0.45

**Semantic Coherence (Current Training):**
- JS divergence: 0.61 (target: <0.3) - needs improvement
- Top-100 overlap: 0.04 (target: >0.5) - needs improvement
- Co-occurrence: 3.39 (target: >20) - needs improvement
- Visit/codes distributions: Match (KS p>0.93)

**Current Focus:** Balancing medical validity with semantic coherence by reducing auxiliary loss weights from (0.01, 0.2) to (0.001, 0.001).

## Known Limitations

1. **Semantic Coherence Issues:** Generated codes are medically valid but statistically implausible (poor co-occurrence patterns)
2. **Architecture Inefficiency:** BART encoder-decoder less efficient than GPT-style decoder-only (but provides richer context)
3. **Medical Invalidity:** ~1-4% of codes remain age/sex-inappropriate despite auxiliary losses
4. **Incomplete Medical Knowledge:** Model lacks ontological understanding (no ICD-9 hierarchy, no SNOMED/UMLS)

See `docs/wiki/08_DEPRECATED_HISTORY.md` for lessons learned from previous approaches.

## Slurm Configuration

Training runs on IllinoisComputes GPU cluster. All Slurm jobs should send email alerts to:
```
jalen.jiang2+slurm@gmail.com
```

## Coding Style

**Language:** Python 3.12

**Standards:**
- Type hints (avoid importing `typing` when possible)
- Google-style docstrings
- Separation of Concerns, Single Responsibility Principle
- No regular expressions unless necessary
- Avoid `abc` class
- Avoid nested functions
- Prioritize simplicity

**No deprecated APIs**

## Communication Style

Direct, blunt, no filler. Proven solutions only. State flaws directly. Skip theory, focus on execution. No emojis, no soft language, no questions. Deliver information and terminate immediately.

## For More Information

- **Getting started:** See `docs/wiki/07_USAGE_GUIDE.md`
- **Understanding architecture:** See `docs/wiki/01_ARCHITECTURE.md`
- **Troubleshooting:** See `docs/wiki/08_DEPRECATED_HISTORY.md`
- **Full documentation:** Browse `docs/wiki/00_INDEX.md`

Important! please remember to commit to a feature branch. Then, once you get to a point where you verify that your changes pass unit tests and you have implemented one idea, please create a pull request with a title and a basic summary of what you changed (like what functions your adding to what files), and why (what purpose does each change serve, what bug you're fixing, what error message). 