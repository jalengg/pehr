# Deprecated History

**Last Updated:** October 24, 2025

This document explains why certain implementations were abandoned and provides a guide to the `deprecated/` directory.

## Timeline of Major Changes

### October 9, 2025: Original Implementation (main.py)

**Approach:** Text-based BART with ICD-9 codes as strings

**Format:**
```
"65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 584.9 <\v> <END>"
```

**Why Deprecated:**

1. **Code Fragmentation Problem**
   - BART tokenizer split ICD-9 codes into subwords
   - Example: `"401.9"` → `["401", ".", "9"]`
   - Result: Model learned subwords, not medical codes
   - Generation output: Gibberish like `"56 ASIAN M <demo> ASIAN m <demoing>"`

2. **Special Token Issues**
   - `<demo>` not registered as special token
   - Fragmented into: `['Ġ<', 'dem', 'o', '>']`
   - Model couldn't learn semantic boundaries

3. **Embedding Initialization**
   - Special tokens had random embeddings (not learned)
   - No gradient flow to custom tokens
   - Model predicted pretrained tokens only

**Lesson Learned:** Medical codes require 1:1 token mapping, not subword tokenization

**Files Deprecated:**
- `deprecated/legacy_implementations/main.py`
- `deprecated/legacy_implementations/main copy.py`

**Documentation:**
- `docs/historical/2025-10-09_tokenization_fix.md`

---

### October 12-17, 2025: Phased Architecture Rebuild

**Phase 1: Data Preparation**

**Changes:**
- Separated demographics from codes
- Created `DiagnosisVocabulary` (1:1 code-to-ID mapping)
- Created `DiagnosisCodeTokenizer` (no fragmentation)

**Why Changed:**
- Fixed code fragmentation issue
- Enabled gradient-based learning of code embeddings

**Files Deprecated:**
- `deprecated/unit_tests/test_phase1.py` (validation tests)

**Documentation:**
- `docs/historical/phase1_data_preparation.md`

---

**Phase 2: Model Architecture**

**Changes:**
- Implemented `ConditionalPrompt` for demographic embeddings
- Extended BART encoder/decoder with prompt injection
- Integrated PromptEHR methodology

**Why Changed:**
- Fixed demographic conditioning weakness
- Matched PromptEHR paper architecture

**Files Deprecated:**
- `deprecated/unit_tests/test_phase2.py` (architecture tests)

**Documentation:**
- `docs/historical/phase2_model_architecture.md`

---

**Phase 3: Dataset & Corruption**

**Changes:**
- Implemented corruption functions (mask infilling, deletion, replacement)
- Created `EHRDataCollator` with sample expansion
- Added next-visit prediction for TPL

**Why Changed:**
- Improved training diversity
- Enabled denoising objectives (BART-style)

**Files Deprecated:**
- `deprecated/unit_tests/test_phase3.py` (dataset tests)
- `deprecated/unit_tests/test_corruptions.py` (corruption function tests)

**Documentation:**
- `docs/historical/phase3_dataset_training.md`

---

### October 19, 2025: Phase 9 - Reparameterization

**Changes:**
- Added d_hidden=128 bottleneck for demographic embeddings
- Implemented dual prompt conditioning (encoder + decoder)
- Offset-based categorical embeddings

**Why Changed:**
- Previous: Weak demographic signal, age/sex violations ~30%
- After: Stronger conditioning, reduced demographic drift

**Why Not Current Default:**
- Further evolution to multi-task learning superseded this

**Files Deprecated:**
- `deprecated/backups/v2/` (pre-reparameterization code)
- `deprecated/unit_tests/test_reparameterization.py`

**Documentation:**
- `docs/historical/phase9_reparameterization.md`

---

### October 23, 2025: Multi-Task Learning

**Changes:**
- Removed race from demographics (medical validity issues)
- Added age/sex prediction heads (token-level)
- Combined loss: `LM + 0.3×age + 0.2×sex` (initial weights)

**Why Changed:**
- Age-inappropriate codes: ~30% → <2%
- Sex-inappropriate codes: ~15% → <1%
- Duplicate codes: High → 0%

**Why Weights Changed Later:**
- Initial weights (0.3, 0.2) destroyed semantic coherence
- Adjusted to (0.01, 0.2), still poor coherence
- Finally reduced to (0.001, 0.001) for balance

**Files Deprecated:**
- `deprecated/backups/v3/` (pre-multi-task code)
- `deprecated/unit_tests/test_token_level.py`

**Documentation:**
- `docs/historical/multitask_implementation_plan.md`
- `docs/reference/multitask_learning.md`

---

### October 24, 2025: Semantic Coherence Fix

**Changes:**
- Reduced auxiliary loss weights: (0.01, 0.2) → (0.001, 0.001)
- Prioritized LM loss over auxiliary tasks

**Why Changed:**
- High auxiliary weights achieved 99% medical validity
- But semantic coherence catastrophically poor:
  - JS divergence: 0.61 (target <0.3)
  - Top-100 overlap: 0.04 (target >0.5)
  - Co-occurrence: 3.39 (target >20)
- Model generated medically valid but statistically implausible code combinations

**Current Status:**
- Training in progress (Job 5501701)
- Expected: Improved semantic coherence, slight medical validity reduction

**Documentation:**
- `docs/analysis/semantic_coherence_fix.md`

---

## Deprecated Utilities

### split_train_test.py

**Purpose:** Split MIMIC-III data into train/test sets

**Why Deprecated:**
- Outputs already generated in `data_splits/`
- Data splits handled in `trainer.py` via `random_split()`
- No longer actively run

**Location:** `deprecated/utilities/split_train_test.py`

---

### analyze_generated.py

**Purpose:** Decode ICD-9 codes to human-readable descriptions

**Why Deprecated:**
- Limited code dictionary (doesn't cover all 5,562 codes)
- Better to use external ICD-9 lookup tools
- Not integrated into evaluation pipeline

**Location:** `deprecated/utilities/analyze_generated.py`

---

### decode_patients.py

**Purpose:** Decode generated patients to human-readable format

**Why Deprecated:**
- Similar functionality to analyze_generated.py
- Evaluation scripts print ICD-9 codes directly
- Manual translation preferred for inspection

**Location:** `deprecated/utilities/decode_patients.py`

---

### translate_codes.py / translate_reconstructions.py

**Purpose:** Translate ICD-9 codes using external mapping

**Why Deprecated:**
- External dependency on code mapping file
- Not integrated into main pipeline
- One-off utility for specific analysis

**Location:**
- `deprecated/utilities/translate_codes.py`
- `deprecated/utilities/translate_reconstructions.py`

---

## Architecture Evolution Summary

```
main.py (text-based)
  ↓ [Fix code fragmentation]
Phase 1: DiagnosisVocabulary + DiagnosisCodeTokenizer
  ↓ [Add demographic conditioning]
Phase 2: ConditionalPrompt + PromptBart
  ↓ [Add data corruption]
Phase 3: EHRDataCollator + corruption functions
  ↓ [Strengthen demographic signal]
Phase 9: Reparameterization + dual prompts
  ↓ [Add medical validity constraints]
Multi-Task: Age/sex prediction heads
  ↓ [Balance validity vs coherence]
Current: Reduced auxiliary loss weights
```

---

## Key Lessons Learned

### 1. Medical Codes ≠ Natural Language

**Problem:** Treating ICD-9 codes as text strings

**Lesson:** Medical codes require:
- 1:1 token mapping (no fragmentation)
- Semantic integrity (code "401.9" is atomic, not "401" + ".")
- Learnable embeddings (not pretrained subwords)

---

### 2. Jaccard Similarity ≠ Semantic Coherence

**Problem:** Using Jaccard as primary quality metric

**Lesson:**
- Jaccard measures exact basket matching
- Doesn't assess clinical plausibility
- Need distributional metrics (JS divergence, co-occurrence)

**Quote from user:**
> "The goal isn't to predict the exact same basket of codes, but to produce semantically probable codes. Jaccard score only measures if we are literally predicting the same basket of codes, but there are so many codes."

---

### 3. Multi-Task Trade-off: Validity vs Coherence

**Problem:** High auxiliary loss weights (0.2-0.3) destroyed semantic coherence

**Lesson:**
- Auxiliary tasks provide weak guidance, not hard constraints
- LM loss must dominate (weights ~0.001 vs 1.0)
- Cannot achieve both perfect validity AND perfect coherence without external knowledge

---

### 4. Architecture Mismatch (Encoder-Decoder vs Decoder-Only)

**Observation:** BART encoder-decoder less efficient than GPT-style decoder-only

**Why Not Changed:**
- Maintains PromptEHR paper compatibility
- Enables bidirectional encoding
- Supports infilling tasks (not just autoregressive)

**Trade-off:** Accepted inefficiency for richer contextual representations

---

## Restoring Old Code

### Restore v3 (Pre-Multi-Task)

```bash
# Backup current code
mkdir current_backup
cp *.py current_backup/

# Restore v3
cp deprecated/backups/v3/*.py .
```

**Use Case:** Compare multi-task vs pure LM performance

---

### Restore v2 (Pre-Reparameterization)

```bash
cp deprecated/backups/v2/*.py .
```

**Use Case:** Test without reparameterization bottleneck

---

### Restore main.py (Original)

```bash
cp deprecated/legacy_implementations/main.py .
```

**Use Case:** Understand original fragmentation issues (for educational purposes)

---

## Future Deprecation Candidates

**Potential candidates for deprecation:**

1. **generate_from_test.py** - Functionality covered by evaluate_medical_validity.py
2. **run_corgan_mimic3.slurm** - If CORGAN baseline no longer needed
3. **train_enhanced.slurm** - If consolidated into train.slurm

**Criteria for Deprecation:**
- Not actively used in last 2 weeks
- Functionality duplicated elsewhere
- No longer compatible with current architecture

---

## Conclusion

**Current Production Code:**
- `trainer.py`, `generate.py`, `evaluate_*.py`
- `config.py`, `data_loader.py`, `dataset.py`
- `prompt_bart_model.py` + components
- `metrics.py`

**Everything Else:** Historical experiments or superseded implementations

**When in Doubt:**
- Check git history for last modification date
- If >2 weeks old and not imported by active code → likely deprecated
- Consult this document for rationale

---

## Navigation

- **Back to Index:** See [Index](00_INDEX.md)
- **Current Architecture:** See [Architecture Overview](01_ARCHITECTURE.md)
- **Usage Guide:** See [Usage Guide](07_USAGE_GUIDE.md)
