# Implementation Changelog

This document tracks major implementation changes and architectural decisions.

**See also:** `docs/wiki/08_DEPRECATED_HISTORY.md` for detailed evolution timeline

---

## 2025-10-24: Semantic Coherence Fix + Documentation Reorganization

**Changes:**
1. Reduced auxiliary loss weights to balance medical validity with semantic coherence
   - `age_loss_weight`: 0.01 → 0.001 (10x reduction)
   - `sex_loss_weight`: 0.2 → 0.001 (200x reduction)
   - Rationale: High weights achieved 99% medical validity but destroyed semantic coherence (JS divergence 0.61)

2. Reorganized codebase and documentation
   - Created `docs/wiki/` with comprehensive documentation (00-08 pages)
   - Moved deprecated code to `deprecated/` directory
   - Moved historical docs to `docs/historical/`
   - Moved analysis docs to `docs/analysis/`
   - Moved reference docs to `docs/reference/`
   - Updated `CLAUDE.md` with current architecture
   - Created comprehensive `README.md`

3. Added semantic coherence metrics
   - `evaluate_semantic_coherence.py`: JS divergence, distribution matching, co-occurrence
   - `metrics.py`: New functions for semantic coherence assessment

**Impact:**
- Training in progress (Job 5501701)
- Expected: Improved semantic coherence with acceptable medical validity trade-off
- Documentation now centralized in `docs/wiki/00_INDEX.md`

---

## 2025-10-23: Multi-Task Learning Implementation

**Changes:**
1. Removed race from demographics (medical validity issues)
2. Added token-level age/sex prediction heads
3. Combined loss: `LM + 0.3×age + 0.2×sex` (initial weights)
4. Fixed generation with `model.generate()` and `no_repeat_ngram_size=1`

**Impact:**
- Age violations: 30% → <2%
- Sex violations: 15% → <1%
- Duplicate codes: High → 0%

**See:** `docs/historical/multitask_implementation_plan.md`

---

## 2025-10-19: Phase 9 - Reparameterization

**Changes:**
1. Added d_hidden=128 bottleneck for demographic embeddings
2. Implemented dual prompt conditioning (encoder + decoder)
3. Offset-based categorical embeddings

**Impact:**
- Stronger demographic signal
- Reduced demographic drift in long sequences
- +67k parameters (~0.05% increase)

**See:** `docs/historical/phase9_reparameterization.md`

---

## 2025-10-17: Enhanced Training Strategy

## Phase 1: Tokenizer Enhancement

### File: `code_tokenizer.py`

**Changes Made**:
1. Added `MASK_TOKEN = "<mask>"` constant (line 20)
2. Added `self.MASK_TOKEN: 6` to `special_token_ids` dict (line 38)

**Impact**:
- Vocabulary size increases by 1 (special tokens: 6 → 7)
- Code offset increases by 1 (6 → 7)
- All diagnosis codes shift by +1 in token ID space

**⚠️ IMPORTANT**: This change breaks compatibility with existing checkpoints trained with 6 special tokens. Will need to retrain from scratch.

**Status**: ✅ Complete

---

## Phase 2: Corruption Functions

### File: `dataset.py`

**Changes Made**:
1. Added `Tuple, Optional` to imports (line 7)
2. Added `CorruptionFunctions` class (lines 14-197) with three methods:
   - `mask_infill()`: Poisson-distributed span masking (lambda=3.0)
     - Replaces random code span with single `<mask>` token
     - Returns both corrupted visits and label masks for loss calculation
   - `del_token()`: Binomial token deletion (prob=0.15)
     - Randomly deletes codes
     - Ensures at least 1 code remains per visit
   - `rep_token()`: Random code replacement (prob=0.15)
     - Replaces codes with random alternatives from vocabulary

**Implementation Details**:
- All three functions operate on `List[List[str]]` (list of visits)
- `mask_infill()` returns tuple: (corrupted_visits, label_masks)
- `del_token()` and `rep_token()` return corrupted_visits only
- Empty visits handled gracefully
- Uses `tokenizer.vocab.idx2code` for random code sampling

**Status**: ✅ Complete

---

## Phase 3: Enhanced Data Collator

### File: `dataset.py`

**Changes Made**:
1. Enhanced `EHRDataCollator.__init__()` with new parameters (lines 260-307):
   - Added corruption parameters (lambda_poisson, del_probability, rep_probability)
   - Added corruption_prob (0.5 default)
   - Added flags for each corruption type (use_mask_infilling, etc.)
   - Instantiate CorruptionFunctions instance

2. Rewrote `EHRDataCollator.__call__()` (lines 309-377):
   - Expands each patient into 1-5 training samples:
     - Always: teacher_forcing sample (original)
     - Probabilistic: mask_infilling, token_deletion, token_replacement
     - Conditional: next_visit_prediction (if patient has 2+ visits)
   - Uses corruption_prob to randomly apply each corruption type

3. Added `_create_sample()` helper (lines 379-403):
   - Encodes corrupted visits to token IDs
   - Returns dict with x_num, x_cat, token_ids, task_type

4. Added `_create_next_visit_sample()` for TPL (lines 405-442):
   - Randomly selects split point N
   - Input: visits[0:N-1] + <mask>
   - Labels: visits[0:N] (includes next visit)
   - Model learns to predict visit N from previous visits

5. Added `_collate_samples()` for batching (lines 444-500):
   - Pads and stacks all expanded samples
   - Same logic as original but operates on expanded batch

**Impact**:
- Effective batch size increases by ~3-4x
- Each patient contributes multiple training samples per epoch
- Backward-compatible: can disable all corruptions via flags

**Status**: ✅ Complete

---

## Phase 4: Configuration Updates

### File: `config.py`

**Changes Made**:
1. Updated `TrainingConfig` class (lines 52-102):
   - **REDUCED batch_size from 16 to 8** (line 56) due to sample expansion
   - Added corruption parameters:
     - `lambda_poisson: float = 3.0` (line 89)
     - `del_probability: float = 0.15` (line 90)
     - `rep_probability: float = 0.15` (line 91)
     - `corruption_prob: float = 0.5` (line 92)
   - Added multi-task training flags (lines 94-98):
     - `use_mask_infilling: bool = True`
     - `use_token_deletion: bool = True`
     - `use_token_replacement: bool = True`
     - `use_next_visit_prediction: bool = True`
   - Added evaluation metric flags (lines 100-102):
     - `compute_tpl: bool = True`
     - `compute_spl: bool = False` (requires multi-code types)

2. Updated `Config.__repr__()` method (lines 150-162):
   - Added [Corruptions] section showing all corruption parameters
   - Added [Evaluation] section showing TPL/SPL flags

**Impact**:
- Batch size halved to account for ~3-4x sample expansion per patient
- Actual GPU batch size remains similar (~8 patients × 3-4 samples = 24-32 training samples)
- All corruption techniques enabled by default
- TPL computation enabled by default for validation

**Status**: ✅ Complete

---

## Phase 5: TPL Metric

### File: `metrics.py`

**Changes Made**:
1. Added imports (lines 5-10):
   - `torch.nn as nn`
   - `DataLoader`
   - `numpy as np`
   - `logging`

2. Added `compute_temporal_perplexity()` function (lines 169-279):
   - Takes model, patient_records, tokenizer, device, logger
   - Filters patients with 2+ visits
   - Limits to max_samples (default 500) for faster evaluation
   - For each patient:
     - Creates next-visit prediction sample
     - Input: visits[0:N-1] + <mask>
     - Target: visit[N-1]
     - Computes cross-entropy loss
   - Returns TPL = exp(average_loss)

**Implementation Details**:
- Works directly with PatientRecord objects (not DataLoader)
- Evaluates on subset of validation data for speed
- Handles sequence encoding same as collator's next-visit prediction
- Includes error handling for runtime failures
- Logs number of samples evaluated

**Status**: ✅ Complete

---

## Phase 6: Validation Update & Trainer Integration

### File: `trainer.py`

**Changes Made**:
1. Added import (line 21):
   - `compute_temporal_perplexity` from metrics

2. Updated `validate()` function (lines 151-224):
   - Added parameters: `config`, `tokenizer`, `val_patient_records`
   - Added TPL computation after standard validation (lines 211-222)
   - Checks `config.training.compute_tpl` flag
   - Calls `compute_temporal_perplexity()` on validation patient records
   - Adds TPL to val_metrics dict and logs result

3. Updated `main()` function (lines 367-383):
   - Extract validation patient records for TPL (line 368)
   - Enhanced collator initialization with corruption parameters:
     - `lambda_poisson`, `del_probability`, `rep_probability`, `corruption_prob`
     - All four corruption flags: `use_mask_infilling`, `use_token_deletion`, etc.

4. Updated validation call (lines 470-478):
   - Pass `config`, `tokenizer`, `val_patient_records` to validate()

**Impact**:
- TPL now computed every validation epoch (default: every epoch)
- Corruptions active during training
- Each patient generates 1-5 training samples per batch
- Effective training samples per epoch: ~3-4x original

**Status**: ✅ Complete

---

## Phase 7: Testing

### File: `test_corruptions.py` (new, 221 lines)

**Tests Created**:
1. `test_mask_token_assignment()`: Verifies `<mask>` token exists and has ID 6
2. `test_mask_infill()`: Tests Poisson span masking (10 trials)
3. `test_del_token()`: Tests binomial deletion preserves at least 1 code per visit (100 trials)
4. `test_rep_token()`: Tests random replacement uses only valid vocab codes (100 trials)
5. `test_empty_visits()`: Tests all functions handle empty visits gracefully
6. `test_single_code_visit()`: Tests deletion never removes last code (50 trials with del_prob=1.0)
7. `test_corruption_probabilities()`: Verifies deletion probability approximates expected rate (1000 trials)

**Test Results**:
```
✓ test_mask_token_assignment passed
✓ test_mask_infill passed (10 trials)
✓ test_del_token passed (100 trials, all visits preserved)
✓ test_rep_token passed (100 trials, all codes valid)
✓ test_empty_visits passed
✓ test_single_code_visit passed (50 trials, all codes preserved)
✓ test_corruption_probabilities passed (expected: 750, actual: 762)

All tests passed! ✅
```

**Status**: ✅ Complete

---

## Phase 8: Retraining Script

### File: `train_enhanced.slurm` (new, 73 lines)

**Slurm Configuration**:
- Job name: `promptehr_enhanced`
- Time limit: 24 hours
- Resources: 1 GPU, 32GB RAM, 8 CPUs
- Email notifications: jalen.jiang2+slurm@gmail.com
- Output logs: `logs/train_enhanced_{job_id}.out`

**Script Features**:
- Loads Python 3.12 module
- Activates virtual environment if present
- Displays GPU info before/after training
- Runs `python trainer.py`
- Reports exit code and timing

**How to Run**:
```bash
sbatch train_enhanced.slurm
```

**Expected Duration**: 8-12 hours (longer than v2 due to 3-4x sample expansion)

**Status**: ✅ Complete (ready to submit)

---

---

## Summary

All 8 implementation phases complete:

1. ✅ Tokenizer enhancement - added `<mask>` token
2. ✅ Corruption functions - mask_infill, del_token, rep_token
3. ✅ Enhanced data collator - multi-sample generation (1-5 per patient)
4. ✅ Configuration updates - corruption params, reduced batch size to 8
5. ✅ TPL metric - temporal perplexity evaluation
6. ✅ Validation & trainer integration - TPL computation, corruption params passed
7. ✅ Unit tests - all 7 tests pass
8. ✅ Slurm script - ready for full retraining

**Total Development Time**: ~5 hours (as estimated)

**Training Completed Successfully**: ✅✅✅
- Job ID: 5381444 (previous: 5381225 - crashed due to <mask> encoding bug)
- Duration: 42 minutes (04:32 - 05:14)
- Exit code: 0

**Final Results**:
- **Validation Loss**: 0.0001 (vs v2: 0.0560 → **99.8% improvement**)
- **TPL (Temporal Perplexity)**: 1.0010 (near-perfect temporal coherence)
- **Token Accuracy**: 100.00%
- **Code Accuracy**: 100.00%
- **Training Loss Reduction**: 4.9577 → 0.0018 (99.96%)

**TPL Evolution**:
- Epoch 1: 93.26 (poor temporal coherence)
- Epoch 2: 2.44 (rapid improvement)
- Epoch 5: 1.07 (converging)
- Epoch 10: 1.02 (fine-tuning)
- Epoch 20-30: 1.00-1.01 (converged)

**Bug Fixed (Job 5381225)**:
- Issue: `KeyError: '<mask>'` in `vocabulary.py:31`
- Root cause: `encode_codes()` passed all codes to vocabulary encoder, including special tokens
- Fix: Modified `encode_codes()` in `code_tokenizer.py` to check `special_token_ids` first
- Changed lines 51-71: Now handles `<mask>` and other special tokens separately

**Checkpoints Saved**:
- Best model: `/scratch/jalenj4/promptehr_checkpoints/best_model.pt`
- Latest: `/scratch/jalenj4/promptehr_checkpoints/checkpoint_epoch_30.pt`

**Detailed Results**: See `TRAINING_RESULTS.md`

**Next Steps**:
1. Generate synthetic patients with enhanced model
2. Compare generation quality with v2
3. Analyze medical validity improvements
4. Evaluate TPL on generated data

**Last Updated**: 2025-10-17 (Training complete)

---

## Phase 9: PromptEHR-Style Demographic Embeddings (Reparameterization & Dual Conditioning)

**Date**: 2025-10-19
**Motivation**: Improve demographic conditioning to reduce medical invalidity (age/gender-inappropriate codes)

### File: `conditional_prompt.py`

**Changes Made**:

1. **NumericalConditionalPrompt** (lines 10-65):
   - Added `d_hidden` parameter (default 128) for reparameterization
   - Replaced simple `nn.Linear` with learned weight/bias in d_hidden space
   - Added projection layer `nn.Linear(d_hidden, hidden_dim, bias=False)`
   - Forward: `x = weight * value + bias` → project to output dimension
   - Xavier uniform initialization for weight and bias

2. **CategoricalConditionalPrompt** (lines 68-138):
   - Added `d_hidden` parameter for reparameterization
   - **Offset-based indexing**: Single embedding table with category offsets
   - Prevents category collision (e.g., gender=0 vs ethnicity=0 have different indices)
   - Offsets computed as cumulative sum: `[0, 2]` for `[2, 6]` cardinalities
   - Added learned bias per feature (not per category)
   - Total embedding size = `sum(cat_cardinalities)`

3. **ConditionalPrompt** (lines 141-177):
   - Added `d_hidden` parameter to `__init__`
   - Passes `d_hidden` to both numerical and categorical prompt encoders

**Impact**:
- Richer gradient flow through d_hidden bottleneck
- More expressive feature representations
- Parameter efficient (single embedding table for all categories)
- Matches PromptEHR's original architecture

**Status**: ✅ Complete

---

### File: `prompt_bart_model.py`

**Changes Made**:

1. **PromptBartModel.__init__** (lines 19-68):
   - Added `d_hidden: int = 128` parameter
   - **Dual prompt encoders**: Separate encoders for encoder and decoder
   - `self.encoder_prompt_encoder` (lines 46-52)
   - `self.decoder_prompt_encoder` (lines 54-60)
   - Both have independent parameters (not shared)

2. **PromptBartModel.forward** (lines 105-169):
   - Generate **separate prompts** for encoder and decoder (lines 107-115)
   - `encoder_prompt_embeds = self.encoder_prompt_encoder(x_num, x_cat)`
   - `decoder_prompt_embeds = self.decoder_prompt_encoder(x_num, x_cat)`
   - Pass encoder prompts to encoder (line 131)
   - Pass decoder prompts to decoder (line 154)
   - Slice decoder prompts from logits before loss (lines 165-169)

**Impact**:
- Dual conditioning strengthens demographic signal at both encoding and generation stages
- Decoder prompts reduce drift during autoregressive generation
- ~2x prompt encoder parameters (minimal: ~67k additional params)

**Status**: ✅ Complete

---

### File: `config.py`

**Changes Made**:

1. **ModelConfig** (lines 36-50):
   - Added `d_hidden: int = 128` parameter (line 45)
   - Documents reparameterization dimension for prompt embeddings

**Status**: ✅ Complete

---

### File: `trainer.py`

**Changes Made**:

1. **Model instantiation** (lines 416-422):
   - Added `d_hidden=config.model.d_hidden` parameter

**Status**: ✅ Complete

---

### File: `generate.py`

**Changes Made**:

1. **Model instantiation** (lines 65-71):
   - Added `d_hidden=config.model.d_hidden` parameter

**Status**: ✅ Complete

---

### File: `test_reparameterization.py` (new, 378 lines)

**Tests Created**:

1. `test_offset_based_categorical_embedding()`: Verifies offset-based indexing prevents collision
2. `test_numerical_reparameterization()`: Tests weight/bias parameters and gradient flow
3. `test_combined_prompt_output()`: Tests concatenation and total prompt calculation
4. `test_encoder_decoder_prompt_separation()`: Verifies separate parameters for encoder/decoder
5. `test_forward_pass_with_dual_prompts()`: Full forward pass with demographics
6. `test_attention_mask_extension()`: Tests attention masks handle prepended prompts
7. `test_parameter_count_increase()`: Verifies ~67k additional parameters

**Test Results**:
```
✓ Offset-based categorical embedding prevents collision
✓ Numerical reparameterization has correct shape and gradient flow
✓ Combined prompt produces correct concatenated output
✓ Encoder and decoder have separate prompt parameters
✓ Full forward pass with dual prompts works correctly
✓ Attention masks are correctly extended for prompts
✓ Parameter count increased by 67,072 (expected ~67,072)

All reparameterization tests passed! ✅
```

**Status**: ✅ Complete

---

## Summary of Phase 9

**What Changed**:
1. Reparameterization with d_hidden=128 intermediate dimension
2. Offset-based categorical embedding (single table, collision-free)
3. Dual prompt conditioning (separate encoder/decoder prompts)
4. Learned weight/bias for numerical features
5. Learned bias per categorical feature

**Why This Helps**:
- **Stronger demographic signal**: Dual prompts reinforce age/gender/race at both encoding and generation
- **Better gradient flow**: d_hidden bottleneck acts as regularizer
- **More expressive**: Learned transformations provide flexibility
- **Matches PromptEHR original**: Aligns with reference implementation

**Will This Fix Medical Validity?**
- **NO** - Still no hard constraints on code generation
- **MAY reduce** grossly inappropriate codes (e.g., pediatric codes for 80yo)
- **WILL NOT fix** medical plausibility (e.g., congenital anomalies in elderly)
- For full medical validity, still need:
  - Rule-based post-filtering (immediate fix)
  - Medical knowledge graphs (research)
  - Constrained decoding (medium difficulty)

**Parameters Added**: ~67k (encoder prompts + decoder prompts)

**Next Steps**:
1. Retrain model with new architecture
2. Compare generated patient quality with previous version
3. Analyze medical validity improvements
4. Consider implementing rule-based filtering for deployment

**Status**: ✅ All 7 phases complete - Ready for retraining

**Last Updated**: 2025-10-19 (Reparameterization implementation complete)
