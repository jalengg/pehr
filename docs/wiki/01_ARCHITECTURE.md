# Architecture Overview

**Last Updated:** October 24, 2025

## System Overview

PromptEHR is a synthetic Electronic Health Record (EHR) generation system based on fine-tuned BART (Bidirectional and Auto-Regressive Transformers) with demographic conditioning. It learns from MIMIC-III clinical data to generate realistic patient sequences containing demographics and medical visit histories with ICD-9 diagnosis codes.

## High-Level Data Flow

```
MIMIC-III Data
    ↓
Data Loader (load_mimic_data)
    ↓
Vocabulary Creation (DiagnosisVocabulary)
    ↓
Tokenization (DiagnosisCodeTokenizer)
    ↓
Dataset + Corruption (EHRPatientDataset, EHRDataCollator)
    ↓
Model Training (PromptBartWithDemographicPrediction)
    ↓
Generation (Conditional or Zero-Prompt)
    ↓
Evaluation (Medical Validity + Semantic Coherence)
```

## Core Components

### 1. Data Processing Layer

**Files:**
- `data_loader.py` - MIMIC-III data loading and preprocessing
- `vocabulary.py` - Diagnosis code vocabulary (1:1 code-to-ID mapping)
- `code_tokenizer.py` - Tokenizer combining special tokens + diagnosis codes
- `dataset.py` - PyTorch dataset with corruption functions

**Purpose:**
Load raw MIMIC-III CSVs, extract patient demographics and visit histories, create vocabulary of 5,562 unique ICD-9 codes, tokenize without fragmentation.

**Key Decision:** Each ICD-9 code = single token ID (no subword fragmentation). This prevents codes like "401.9" from being split into ["401", ".", "9"].

### 2. Model Architecture Layer

**Files:**
- `prompt_bart_model.py` - Main model class with multi-task learning
- `prompt_bart_encoder.py` - Custom BART encoder with prompt injection
- `prompt_bart_decoder.py` - Custom BART decoder with prompt injection
- `conditional_prompt.py` - Demographic embedding with reparameterization

**Purpose:**
Extend BART encoder-decoder with:
1. Demographic conditioning via prompt embeddings injected into every layer
2. Multi-task learning (language modeling + age prediction + sex prediction)
3. Dual prompt conditioning (separate prompts for encoder and decoder)

**Key Decision:** Use BART encoder-decoder architecture (despite being inefficient for causal LM) to maintain compatibility with PromptEHR paper methodology.

### 3. Training Layer

**Files:**
- `trainer.py` - Training loop, optimization, validation
- `config.py` - Hyperparameters and configuration
- `metrics.py` - Loss tracking, perplexity, TPL, Jaccard

**Purpose:**
Train model with multi-task loss (LM + auxiliary age/sex prediction), track metrics during validation, save checkpoints.

**Key Decision:** Reduced auxiliary loss weights to 0.001 (from 0.01/0.2) to prioritize learning realistic code distributions over pure medical validity.

### 4. Generation Layer

**Files:**
- `generate.py` - Conditional and zero-prompt generation functions

**Purpose:**
Generate synthetic patients via:
1. **Conditional generation:** Reconstruct patient from partial code prompts
2. **Zero-prompt generation:** Generate from demographics only (age, sex)

**Key Decision:** Use HuggingFace `model.generate()` with `no_repeat_ngram_size=1` to prevent duplicate codes within visits.

### 5. Evaluation Layer

**Files:**
- `evaluate_medical_validity.py` - Age/sex appropriateness, duplicates
- `evaluate_semantic_coherence.py` - JS divergence, co-occurrence, distribution matching

**Purpose:**
Assess both medical validity (are codes appropriate for demographics?) and semantic coherence (do codes match training distributions?).

**Key Decision:** Separate medical validity from semantic coherence. Jaccard similarity measures exact match, not semantic plausibility.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MIMIC-III Data Files                      │
│  (PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Loader (data_loader.py)               │
│  - Merge patient demographics with admissions/diagnoses      │
│  - Calculate age at first admission                          │
│  - Group visits chronologically per patient                  │
│  Output: PatientRecord(visits, age, gender)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Vocabulary + Tokenizer                          │
│  DiagnosisVocabulary: code ↔ index mapping                   │
│  DiagnosisCodeTokenizer: Add special tokens + codes          │
│  Special tokens: <s>, </s>, <pad>, <unk>, <v>, <\v>, <mask> │
│  Code offset: 7 (codes start at token ID 7)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Dataset (dataset.py)                        │
│  EHRPatientDataset: Stores patient records                   │
│  EHRDataCollator: Batching + corruption                      │
│    - Mask infilling (Poisson λ=3.0)                          │
│    - Token deletion (p=0.15)                                 │
│    - Token replacement (p=0.15)                              │
│    - Next-visit prediction (for TPL)                         │
│  Output: Batches with x_num, x_cat, token_ids, labels       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│      Model (prompt_bart_model.py + components)               │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  ConditionalPrompt (conditional_prompt.py)       │       │
│  │  - Reparameterization (d_hidden=128 bottleneck)  │       │
│  │  - Numerical: weight×age + bias → project to 768 │       │
│  │  - Categorical: embedding + bias → project to 768│       │
│  │  Output: [batch, 1, 768] prompt vectors          │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  PromptBartEncoder (prompt_bart_encoder.py)      │       │
│  │  - Injects prompts into every encoder layer      │       │
│  │  - Extends attention masks for prompt tokens     │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  PromptBartDecoder (prompt_bart_decoder.py)      │       │
│  │  - Injects prompts into every decoder layer      │       │
│  │  - Separate prompt encoder from encoder prompts  │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Prediction Heads                                │       │
│  │  - LM head: Generate next code token             │       │
│  │  - Age head: Predict age category (6 classes)    │       │
│  │  - Sex head: Predict sex (binary)                │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                    │
│                         ↓                                    │
│  Combined Loss = LM + 0.001×Age + 0.001×Sex                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                Training Loop (trainer.py)                    │
│  - AdamW optimizer (lr=1e-4, weight_decay=0.01)              │
│  - Linear warmup (1000 steps)                                │
│  - Gradient clipping (max_norm=1.0)                          │
│  - Validation every epoch                                    │
│  - Checkpoint best model by validation loss                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Generation (generate.py)                        │
│  - Conditional: generate_patient_sequence_conditional()      │
│    Input: target patient, prompt_prob (0.0-1.0)              │
│    Output: Generated visits given partial prompts            │
│  - Zero-prompt: generate_patient_from_demographics()         │
│    Input: age, sex                                           │
│    Output: Fully synthetic patient                           │
│  - Uses model.generate() with temperature, top-k, top-p      │
│  - Duplicate suppression: no_repeat_ngram_size=1             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation                                │
│  evaluate_medical_validity.py:                               │
│    - Age appropriateness (>98% target)                       │
│    - Sex appropriateness (>99% target)                       │
│    - Duplicate rate (<1% target)                             │
│    - Jaccard similarity (0.40-0.45 target)                   │
│                                                              │
│  evaluate_semantic_coherence.py:                             │
│    - JS divergence (code frequencies, <0.3 target)           │
│    - Distribution match (KS tests, p>0.05 target)            │
│    - Top-100 overlap (>0.5 target)                           │
│    - Co-occurrence score (>20 target)                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

### Why BART Encoder-Decoder?

**Decision:** Use BART encoder-decoder architecture instead of GPT-style decoder-only.

**Reason:**
- Maintains compatibility with PromptEHR paper methodology
- Allows bidirectional encoding of demographics and prompt codes
- Enables infilling tasks (not just autoregressive generation)

**Trade-off:** Less efficient than decoder-only for pure causal LM, but provides richer contextual representations.

### Why Separate Demographics from Codes?

**Decision:** Represent demographics as continuous/categorical features (x_num, x_cat) instead of text tokens.

**Reason:**
- Prevents tokenizer fragmentation (age "65" → single value, not ["6", "5"])
- Enables gradient-based learning of demographic embeddings
- Matches PromptEHR's prompt-based conditioning approach

**Alternative rejected:** Text-based sequences like "65 M <demo> <v> 401.9 ..." (caused early tokenization issues)

### Why Dual Prompt Conditioning?

**Decision:** Inject separate demographic prompts into both encoder and decoder.

**Reason:**
- Encoder prompts: Influence contextual representation of input codes
- Decoder prompts: Reinforce demographics during generation, reduce drift

**Alternative rejected:** Encoder-only prompts (demographic drift in long sequences)

### Why Multi-Task Learning?

**Decision:** Add auxiliary age/sex prediction heads with small loss weights (0.001).

**Reason:**
- Encourages model to generate age/sex-appropriate codes
- Provides additional supervision signal
- Low weights (0.001) prevent auxiliary tasks from dominating

**Alternative rejected:** Pure LM (generates medically invalid codes), high weights (0.01-0.2, destroys semantic coherence)

### Why No Repeat N-gram Size?

**Decision:** Use `no_repeat_ngram_size=1` during generation.

**Reason:**
- Prevents duplicate codes within a single visit
- Clinically implausible to diagnose same code twice in one admission

**Alternative rejected:** Post-processing deduplication (affects generation statistics)

## File-to-Component Mapping

| Component | Files | Purpose |
|-----------|-------|---------|
| **Data Loading** | `data_loader.py`, `vocabulary.py` | MIMIC-III preprocessing, vocabulary creation |
| **Tokenization** | `code_tokenizer.py` | Diagnosis code tokenization without fragmentation |
| **Dataset** | `dataset.py` | PyTorch dataset, batching, corruption functions |
| **Demographic Conditioning** | `conditional_prompt.py` | Reparameterization, prompt embedding generation |
| **Model Core** | `prompt_bart_model.py`, `prompt_bart_encoder.py`, `prompt_bart_decoder.py` | BART with dual prompt injection + multi-task heads |
| **Training** | `trainer.py`, `config.py` | Training loop, optimization, configuration |
| **Metrics** | `metrics.py` | Loss tracking, perplexity, TPL, reconstruction Jaccard |
| **Generation** | `generate.py` | Conditional and zero-prompt generation |
| **Evaluation** | `evaluate_medical_validity.py`, `evaluate_semantic_coherence.py` | Medical validity and semantic coherence assessment |

## Next Steps

- **Understand data processing:** See [Data Pipeline](02_DATA_PIPELINE.md)
- **Deep dive into model:** See [Model Architecture](03_MODEL_ARCHITECTURE.md)
- **Learn how to train:** See [Training](04_TRAINING.md)
- **Generate patients:** See [Generation](05_GENERATION.md)
- **Evaluate quality:** See [Evaluation](06_EVALUATION.md)
