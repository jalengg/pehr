# 39: File Reference - Complete Codebase Map

**Estimated Time:** 45 minutes (reference, not sequential reading)
**Prerequisites:** None (use as reference throughout learning)
**Next:** [40_COMMON_TASKS.md](40_COMMON_TASKS.md)

---

## Purpose

This page provides a **complete file-by-file reference** for the PromptEHR codebase. Use this to:
- Quickly find which file implements what functionality
- Understand import dependencies
- Locate specific components when debugging
- Plan modifications to the codebase

**Pro Tip:** Keep this page open while coding. Use Ctrl+F to search for functionality.

---

## Quick Lookup Table

| Need to... | File | Key Functions |
|------------|------|---------------|
| Load MIMIC-III data | `data_loader.py` | `load_mimic_data()` |
| Build vocabulary | `vocabulary.py` | `DiagnosisVocabulary.build_vocab()` |
| Tokenize codes | `code_tokenizer.py` | `DiagnosisCodeTokenizer.encode_codes()` |
| Create training dataset | `dataset.py` | `EHRPatientDataset` |
| Define model architecture | `prompt_bart_model.py` | `PromptBartWithDemographicPrediction` |
| Train model | `trainer.py` or `trainer_hierarchical.py` | `main()` |
| Generate patients | `generate.py` | `generate_sequences()` |
| Evaluate medical validity | `evaluate_medical_validity.py` | `evaluate_medical_validity()` |
| Evaluate semantic coherence | `evaluate_semantic_coherence.py` | `evaluate_semantic_coherence()` |
| Build ICD-9 hierarchy | `icd9_hierarchy.py` | `ICD9Hierarchy` |
| Hierarchical tokenization | `hierarchical_tokenizer.py` | `HierarchicalDiagnosisTokenizer` |
| Two-stage generation | `hierarchical_generation.py` | `generate_hierarchical()` |
| Co-occurrence matrix | `cooccurrence_utils.py` | `build_cooccurrence_matrix()` |
| Track metrics | `metrics.py` | `MetricsTracker` |
| Configure hyperparameters | `config.py` | All config dataclasses |

---

## File Organization by Category

### 📁 **Data Processing**

#### `data_loader.py` (350 lines)
**Purpose:** Load and preprocess MIMIC-III data

**Key Classes:**
- `PatientRecord`: Container for patient data (age, gender, visits)

**Key Functions:**
- `load_mimic_data(data_dir, n_patients, max_seq_len)` - Main loader
- `normalize_ethnicity(ethnicity)` - Ethnicity categorization (deprecated, race removed)

**Important Details:**
- Loads 3 CSV files: PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv
- Joins on SUBJECT_ID and HADM_ID
- Groups diagnoses by admission → visits
- Calculates age from DOB and admission time
- Gender encoding: 0=M, 1=F

**File Location:** `/u/jalenj4/pehr_scratch/data_loader.py:1-350`

---

#### `vocabulary.py` (200 lines)
**Purpose:** Diagnosis code vocabulary management

**Key Class:**
- `DiagnosisVocabulary`: 1:1 code → token ID mapping

**Key Methods:**
- `build_vocab(patients)` - Extract unique codes from patient records
- `add_code(code)` - Add single code to vocabulary
- `get_code(idx)` - Get code from token ID
- `get_idx(code)` - Get token ID from code

**Vocabulary Structure:**
```python
vocab: List[str] = ["401.9", "250.00", "428.0", ...]  # 6,985 codes
code2idx: Dict[str, int] = {"401.9": 0, "250.00": 1, ...}
idx2code: Dict[int, str] = {0: "401.9", 1: "250.00", ...}
```

**File Location:** `/u/jalenj4/pehr_scratch/vocabulary.py:1-200`

---

#### `code_tokenizer.py` (300 lines)
**Purpose:** Custom tokenizer for diagnosis codes (no fragmentation)

**Key Class:**
- `DiagnosisCodeTokenizer`: Extends vocabulary with special tokens

**Special Tokens (7 total):**
```python
<s>     (BOS) = 0
<pad>   = 1
</s>    (EOS) = 2
<unk>   = 3
<v>     (visit start) = 4
<\v>    (visit end) = 5
<mask>  = 6
```

**Code Tokens:** Start at offset 7
- "401.9" → token ID 7
- "250.00" → token ID 8
- ...

**Key Methods:**
- `encode_codes(codes)` - List of codes → List of token IDs
- `decode_codes(token_ids)` - List of token IDs → List of codes
- `encode_patient(patient)` - Full patient → token sequence

**File Location:** `/u/jalenj4/pehr_scratch/code_tokenizer.py:1-300`

---

#### `dataset.py` (600 lines)
**Purpose:** PyTorch dataset with data corruption

**Key Classes:**
- `EHRPatientDataset`: Main dataset for training
- `EHRDataCollator`: Batch collation with corruption

**Corruption Functions:**
- `corrupt_sequence()` - Main corruption dispatcher
- `mask_infill()` - Poisson(λ=3) span masking
- `token_deletion()` - 15% deletion probability
- `token_replacement()` - 15% replacement probability
- `next_visit_prediction()` - Temporal prediction task

**Corruption Flags:**
```python
use_mask_infilling: bool = True
use_token_deletion: bool = True
use_token_replacement: bool = True
use_next_visit_prediction: bool = True
corruption_prob: float = 0.5  # Apply corruption 50% of time
```

**File Location:** `/u/jalenj4/pehr_scratch/dataset.py:1-600`

---

### 📁 **Model Architecture**

#### `prompt_bart_model.py` (500 lines)
**Purpose:** Main model classes

**Key Classes:**

**1. `PromptBartModel` (lines 16-150):**
- Base BART with prompt conditioning
- Dual encoder/decoder prompts
- No auxiliary tasks

**2. `PromptBartWithDemographicPrediction` (lines 153-350):**
- Extends PromptBartModel
- Adds age prediction head (token-level, MSE loss)
- Adds sex prediction head (token-level, cross-entropy loss)
- Multi-task loss: LM + 0.001×age + 0.001×sex

**Key Components:**
```python
# Age prediction
self.age_head = nn.Linear(config.d_model, 1)

# Sex prediction
self.sex_head = nn.Linear(config.d_model, 2)

# Loss weights
self.age_loss_weight = 0.001
self.sex_loss_weight = 0.001
```

**File Location:** `/u/jalenj4/pehr_scratch/prompt_bart_model.py:1-500`

---

#### `conditional_prompt.py` (250 lines)
**Purpose:** Demographic embedding with reparameterization

**Key Class:**
- `ConditionalPrompt`: Converts demographics → prompt embeddings

**Architecture:**
```python
# Numerical (age)
x_num → Linear(d_hidden=128) → ReLU → Linear(d_model=768)

# Categorical (sex)
x_cat → Embedding → Linear(d_hidden=128) → ReLU → Linear(d_model=768)

# Combined
[num_embeds || cat_embeds] → [batch, n_features, 768]
```

**Why reparameterization?**
- Bottleneck (d_hidden=128) prevents overfitting to demographic combinations
- Learned feature interactions in hidden layer
- Matches PromptEHR paper architecture

**File Location:** `/u/jalenj4/pehr_scratch/conditional_prompt.py:1-250`

---

#### `prompt_bart_encoder.py` (200 lines)
**Purpose:** BART encoder with prompt injection

**Key Class:**
- `PromptBartEncoder`: Extends BartEncoder

**Prompt Injection Mechanism:**
```python
# 1. Embed input tokens
inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len, 768]

# 2. Prepend prompt embeddings
if encoder_prompt is not None:
    inputs_embeds = torch.cat([encoder_prompt, inputs_embeds], dim=1)
    # Result: [batch, n_prompts+seq_len, 768]

# 3. Extend attention mask
prompt_mask = torch.ones(batch_size, n_prompts)
attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
```

**File Location:** `/u/jalenj4/pehr_scratch/prompt_bart_encoder.py:1-200`

---

#### `prompt_bart_decoder.py` (250 lines)
**Purpose:** BART decoder with prompt injection

**Key Class:**
- `PromptBartDecoder`: Extends BartDecoder

**Similar to encoder but for decoder side:**
- Prepends decoder prompt to decoder input embeddings
- Cross-attends to encoder outputs (which include encoder prompts)

**File Location:** `/u/jalenj4/pehr_scratch/prompt_bart_decoder.py:1-250`

---

### 📁 **Training**

#### `trainer.py` (500 lines)
**Purpose:** Flat code training pipeline

**Key Functions:**
- `main()` - Setup and training orchestration
- `train_epoch()` - Single epoch training loop
- `validate_epoch()` - Validation with Jaccard computation

**Training Configuration (from config.py):**
```python
n_patients: 25000
batch_size: 8
epochs: 30
learning_rate: 1e-4
warmup_steps: 1000
```

**Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)
**Scheduler:** Linear warmup → constant

**File Location:** `/u/jalenj4/pehr_scratch/trainer.py:1-500`

---

#### `trainer_hierarchical.py` (505 lines) 🆕
**Purpose:** Hierarchical training pipeline (category-level)

**Differences from trainer.py:**
- Uses `HierarchicalDiagnosisTokenizer` (7,935 vocab)
- Uses `HierarchicalEHRDataset` (category sequences)
- Uses `ICD9Hierarchy` for category mapping
- Integrates co-occurrence loss at code level

**Training Loss:**
```python
total_loss = lm_loss + (0.001 * age_loss) + (0.001 * sex_loss) + (0.05 * cooccur_loss)
```

**Current Training:** Job 5755517 (Nov 11-12, 2025)

**File Location:** `/u/jalenj4/pehr_scratch/trainer_hierarchical.py:1-505`

---

#### `config.py` (150 lines)
**Purpose:** All hyperparameter configuration

**Key Dataclasses:**
```python
@dataclass
class DataConfig:
    n_patients: int = 25000
    max_seq_len: int = 256
    train_val_split: float = 0.2

@dataclass
class ModelConfig:
    base_model: str = "facebook/bart-base"
    continuous_features: int = 1  # age
    categorical_features: int = 1  # sex
    prompt_length: int = 1
    age_loss_weight: float = 0.001
    sex_loss_weight: float = 0.001
    cooccurrence_loss_weight: float = 0.05  # For hierarchical

@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    warmup_steps: int = 1000

@dataclass
class CorruptionConfig:
    lambda_poisson: float = 3.0
    deletion_prob: float = 0.15
    replacement_prob: float = 0.15
    corruption_prob: float = 0.5
```

**File Location:** `/u/jalenj4/pehr_scratch/config.py:1-150`

---

#### `metrics.py` (250 lines)
**Purpose:** Training metrics tracking

**Key Class:**
- `MetricsTracker`: Accumulates and computes metrics

**Tracked Metrics:**
```python
losses: List[float]            # Total combined loss
perplexities: List[float]      # exp(lm_loss)
token_accuracies: List[float]  # All tokens
code_accuracies: List[float]   # Diagnosis codes only
lm_losses: List[float]         # Language modeling
age_losses: List[float]        # Age prediction
sex_losses: List[float]        # Sex prediction
cooccur_losses: List[float]    # Co-occurrence regularization
reconstruction_jaccards: List[float]  # Validation only
```

**Key Methods:**
- `update(...)` - Add batch metrics
- `get_average_metrics()` - Epoch-level averages
- `reset()` - Clear for new epoch

**File Location:** `/u/jalenj4/pehr_scratch/metrics.py:1-250`

---

### 📁 **Generation**

#### `generate.py` (400 lines)
**Purpose:** Patient generation (flat codes)

**Key Functions:**

**1. `generate_sequences()`**
```python
def generate_sequences(
    model, tokenizer, patients,
    demographics_only=False,  # Zero-prompt vs conditional
    num_samples=100
):
    """
    Generate synthetic patient sequences.

    Args:
        demographics_only: If True, zero-prompt (harder task)
                          If False, conditional (partial code prompts)
    """
```

**Two Generation Modes:**
- **Conditional:** Start with partial codes from real patient (reconstruction)
- **Zero-prompt:** Start with only demographics (fully synthetic)

**File Location:** `/u/jalenj4/pehr_scratch/generate.py:1-400`

---

### 📁 **Evaluation**

#### `evaluate_medical_validity.py` (500 lines)
**Purpose:** Medical validity assessment

**Evaluated Metrics:**
1. **Age-Appropriate Rate:**
   - Pediatric codes (V20.x, V21.x): Age < 18
   - Geriatric codes: Age > 65
   - Pregnancy codes (630-679): Ages 15-50

2. **Sex-Appropriate Rate:**
   - Male-only: Prostate (600-608), testicular
   - Female-only: Pregnancy (630-679), ovarian, cervical

3. **Duplicate Rate:**
   - Same code appearing multiple times in one visit

**Key Function:**
```python
def evaluate_medical_validity(synthetic_patients, rules):
    """
    Returns:
        {
            'age_appropriate': 0.99,  # 99% valid
            'sex_appropriate': 0.96,  # 96% valid
            'duplicate_rate': 0.00    # 0% duplicates
        }
    """
```

**Rules File:** `docs/reference/medical_validity.md`

**File Location:** `/u/jalenj4/pehr_scratch/evaluate_medical_validity.py:1-500`

---

#### `evaluate_semantic_coherence.py` (600 lines)
**Purpose:** Semantic coherence assessment

**Evaluated Metrics:**

1. **JS Divergence** (Jensen-Shannon)
   - Measures code frequency distribution similarity
   - Target: <0.3 (lower = better)

2. **Top-100 Overlap**
   - Overlap between top 100 most frequent codes
   - Target: >0.5 (higher = better)

3. **Co-occurrence Score**
   - Average observed pair count per code
   - Target: >20 (higher = better)

4. **KS Tests** (Kolmogorov-Smirnov)
   - Visits per patient distribution
   - Codes per visit distribution
   - Target: p-value >0.05 (not significantly different)

**Key Function:**
```python
def evaluate_semantic_coherence(real_patients, synthetic_patients):
    """
    Returns:
        {
            'js_divergence': 0.61,      # Current: 0.61, target <0.3
            'top100_overlap': 0.04,     # Current: 0.04, target >0.5
            'cooccurrence_score': 2.80, # Current: 2.80, target >20
            'ks_visits_pvalue': 0.93,   # p>0.05: distributions match
            'ks_codes_pvalue': 0.94
        }
    """
```

**File Location:** `/u/jalenj4/pehr_scratch/evaluate_semantic_coherence.py:1-600`

---

### 📁 **Advanced Features (Hierarchical)** 🆕

#### `icd9_hierarchy.py` (200 lines)
**Purpose:** ICD-9 hierarchical structure management

**Key Class:**
- `ICD9Hierarchy`: Extracts and manages 943 categories from 6,985 codes

**Key Methods:**
- `get_category(code)` - "401.9" → "401"
- `get_codes(category)` - "401" → ["401.0", "401.1", "401.9"]
- `get_num_categories()` - 943
- `get_num_codes()` - 6,985

**Statistics:**
- Average codes per category: 7.4
- Sparsity reduction: 7.4x
- Co-occurrence coverage: 1.4% → 26% (18.6x improvement)

**File Location:** `/u/jalenj4/pehr_scratch/icd9_hierarchy.py:1-200`

---

#### `hierarchical_tokenizer.py` (350 lines) 🆕
**Purpose:** Dual vocabulary tokenizer (categories + codes)

**Key Class:**
- `HierarchicalDiagnosisTokenizer`: Extends DiagnosisCodeTokenizer

**Vocabulary Structure:**
```python
Token IDs:
  0-6:    Special tokens (<s>, <pad>, </s>, <unk>, <v>, <\v>, <mask>)
  7-949:  Category tokens (943 categories)
  950-7934: Code tokens (6,985 codes)

Total: 7,935 tokens
```

**Key Offsets:**
- `category_offset = 7`
- `code_offset = 950`

**File Location:** `/u/jalenj4/pehr_scratch/hierarchical_tokenizer.py:1-350`

---

#### `hierarchical_dataset.py` (300 lines) 🆕
**Purpose:** Dataset for category-level training

**Key Class:**
- `HierarchicalEHRDataset`: Converts patient codes → categories

**Process:**
1. Load patient with specific codes: [401.9, 250.00, 428.0]
2. Map to categories: [401, 250, 428]
3. Apply corruption to category sequence
4. Return category token IDs for training

**File Location:** `/u/jalenj4/pehr_scratch/hierarchical_dataset.py:1-300`

---

#### `hierarchical_generation.py` (400 lines) 🆕
**Purpose:** Two-stage generation (categories → codes)

**Key Function:**
- `generate_hierarchical()` - Main two-stage generation

**Two Stages:**

**Stage 1:** Generate category sequence
```python
# Input: Demographics (age, sex)
# Output: Category sequence [401, 250, 428]
```

**Stage 2:** Expand categories → specific codes
```python
# For each category:
#   - Sample codes from hierarchy (frequency-weighted)
#   - Filter by demographics (age/sex rules)
# Output: [401.9, 250.00, 428.0]
```

**File Location:** `/u/jalenj4/pehr_scratch/hierarchical_generation.py:1-400`

---

#### `cooccurrence_utils.py` (350 lines) 🆕
**Purpose:** Co-occurrence matrix and loss computation

**Key Functions:**

**1. `build_cooccurrence_matrix()`**
```python
def build_cooccurrence_matrix(patients, tokenizer, code_offset=7):
    """
    Build sparse co-occurrence matrix.

    Returns:
        scipy.sparse.csr_matrix [vocab_size, vocab_size]
    """
```

**Statistics:**
- Observed pairs: 696,425
- Possible pairs: 48,765,225
- Coverage: 1.4%

**2. `cooccurrence_loss_efficient()`**
```python
def cooccurrence_loss_efficient(generated_code_ids, cooccur_matrix, tokenizer, threshold=5):
    """
    Compute loss penalizing rare code pairs.

    Args:
        threshold: Pairs with count < threshold are penalized

    Returns:
        scalar loss
    """
```

**Loss Weight:** 0.05 (in hierarchical training)

**File Location:** `/u/jalenj4/pehr_scratch/cooccurrence_utils.py:1-350`

---

### 📁 **Testing & Utilities**

#### Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_unconditional.py` | Quick generation test (10 patients) | ✅ Active |
| `test_cooccurrence.py` | Co-occurrence matrix tests | ✅ Active |
| `test_hierarchy.py` | ICD-9 hierarchy tests | ✅ Active |
| `test_hierarchy_full.py` | Full hierarchy integration | ✅ Active |
| `test_hierarchical_tokenizer.py` | Tokenizer tests | ✅ Active |
| `test_hierarchical_dataset.py` | Dataset tests | ✅ Active |
| `test_hierarchical_model.py` | Model integration tests | ✅ Active |
| `test_hierarchical_generation.py` | Two-stage generation tests | ✅ Active |

**Usage:**
```bash
# Run specific test
python test_hierarchy.py

# Run all tests matching pattern
python -m pytest test_hierarchical_*.py
```

---

## Import Dependency Graph

**Simple Dependency Flow:**

```
data_loader.py
    ↓
vocabulary.py
    ↓
code_tokenizer.py
    ↓
dataset.py
    ↓
config.py
    ↓
conditional_prompt.py
prompt_bart_encoder.py
prompt_bart_decoder.py
    ↓
prompt_bart_model.py
    ↓
metrics.py
    ↓
trainer.py
generate.py
evaluate_medical_validity.py
evaluate_semantic_coherence.py
```

**Hierarchical Extensions:**

```
vocabulary.py
    ↓
icd9_hierarchy.py
    ↓
hierarchical_tokenizer.py
hierarchical_dataset.py
    ↓
cooccurrence_utils.py
    ↓
hierarchical_generation.py
trainer_hierarchical.py
```

**Minimal Import Set (for model loading):**
```python
from vocabulary import DiagnosisVocabulary
from code_tokenizer import DiagnosisCodeTokenizer
from prompt_bart_model import PromptBartWithDemographicPrediction
import torch
```

---

## File Sizes & Line Counts

| File | Lines | Size | Complexity |
|------|-------|------|------------|
| `data_loader.py` | 350 | 12 KB | Low |
| `vocabulary.py` | 200 | 6 KB | Low |
| `code_tokenizer.py` | 300 | 10 KB | Medium |
| `dataset.py` | 600 | 22 KB | High |
| `conditional_prompt.py` | 250 | 9 KB | Medium |
| `prompt_bart_encoder.py` | 200 | 8 KB | Medium |
| `prompt_bart_decoder.py` | 250 | 9 KB | Medium |
| `prompt_bart_model.py` | 500 | 18 KB | High |
| `trainer.py` | 500 | 18 KB | High |
| `trainer_hierarchical.py` | 505 | 19 KB | High |
| `generate.py` | 400 | 15 KB | Medium |
| `metrics.py` | 250 | 9 KB | Low |
| `config.py` | 150 | 5 KB | Low |
| `evaluate_medical_validity.py` | 500 | 18 KB | Medium |
| `evaluate_semantic_coherence.py` | 600 | 22 KB | High |
| `icd9_hierarchy.py` | 200 | 7 KB | Low |
| `hierarchical_tokenizer.py` | 350 | 13 KB | Medium |
| `hierarchical_dataset.py` | 300 | 11 KB | Medium |
| `hierarchical_generation.py` | 400 | 15 KB | High |
| `cooccurrence_utils.py` | 350 | 13 KB | High |
| **Total Active** | **6,955** | **~258 KB** | |

---

## Where to Find Specific Functionality

### Model Components
- Prompt conditioning: `conditional_prompt.py:20-120`
- Encoder prompt injection: `prompt_bart_encoder.py:80-120`
- Decoder prompt injection: `prompt_bart_decoder.py:90-130`
- Age prediction head: `prompt_bart_model.py:180-200`
- Sex prediction head: `prompt_bart_model.py:202-220`
- Combined loss: `prompt_bart_model.py:240-280`

### Data Processing
- MIMIC-III loading: `data_loader.py:100-250`
- Vocabulary building: `vocabulary.py:50-100`
- Code tokenization: `code_tokenizer.py:80-150`
- Data corruption: `dataset.py:200-500`
- Span masking: `dataset.py:250-300`

### Training
- Training loop: `trainer.py:100-250` or `trainer_hierarchical.py:100-250`
- Validation: `trainer.py:260-400`
- Metrics tracking: `metrics.py:50-200`
- Checkpoint saving: `trainer.py:450-480`

### Generation
- Flat generation: `generate.py:50-200`
- Hierarchical generation: `hierarchical_generation.py:50-250`
- Two-stage expansion: `hierarchical_generation.py:260-400`

### Evaluation
- Medical validity: `evaluate_medical_validity.py:100-400`
- Semantic coherence: `evaluate_semantic_coherence.py:100-500`
- Co-occurrence analysis: `evaluate_semantic_coherence.py:300-400`

---

## Key Takeaways

1. **29 active Python files** (7,000 lines total) organized into clear categories

2. **4 major categories:** Data processing, model architecture, training/evaluation, advanced features

3. **Hierarchical files (8 files)** are all **🆕 novel contributions** (not in PromptEHR codebase)

4. **Two training pipelines:** `trainer.py` (flat) and `trainer_hierarchical.py` (category-level)

5. **Import flow:** Linear from data → vocab → tokenizer → dataset → model → training/generation

6. **Use this page as reference** when navigating codebase or debugging

---

## What's Next?

You now have a **complete file reference** for the entire codebase.

**Next:** [40_COMMON_TASKS.md](40_COMMON_TASKS.md) - Learn how to perform common tasks (training, generation, evaluation) with copy-paste commands.

**Alternative Path:**
- [03_OOP_STRUCTURE.md](03_OOP_STRUCTURE.md) - Understand class relationships
- Search this page (Ctrl+F) for specific functionality

---

**Navigation:**
- ← Back to [38_DEBUGGING_GUIDE.md](38_DEBUGGING_GUIDE.md)
- → Next: [40_COMMON_TASKS.md](40_COMMON_TASKS.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
