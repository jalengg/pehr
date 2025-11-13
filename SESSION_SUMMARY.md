# Session Summary: Co-occurrence Learning & Hierarchical Generation

**Date:** 2025-11-05
**Branch:** `feature/cooccurrence-learning`
**Goal:** Improve semantic coherence through co-occurrence regularization and hierarchical generation

---

## Completed Work

### 1. Co-occurrence Regularization ✅

**Implementation:**
- `cooccurrence_utils.py` (415 lines): Build pairwise code co-occurrence matrix and compute regularization loss
- `config.py`: Added `cooccurrence_loss_weight: 0.05`, increased `num_patients: 50000`
- `trainer.py`: Integrated co-occurrence loss into training loop
- `test_cooccurrence.py`: Integration test (all passed)

**Results:**
```
Training data: 50k patients → 25k for matrix building
Co-occurrence matrix: 6,985 × 6,985 codes
Coverage: 3.25% of possible code pairs
Loss: Penalizes rare pairs (threshold=5 co-occurrences)
```

**Integration:**
```python
total_loss = lm_loss +
             0.005 * age_loss +
             0.005 * sex_loss +
             0.05 * cooccur_loss
```

### 2. Documentation ✅

**Files Created:**
- `docs/analysis/semantic_coherence_research.md`: Research on semantic coherence solutions
- `docs/reference/original_promptehr_implementation.md`: Complete PromptEHR codebase analysis
- `docs/diagrams/promptehr_vs_ours_architecture.md`: Architecture comparison diagrams

**Key Finding:**
- PromptEHR evaluates **perplexity only** (no semantic coherence)
- Our 7 semantic coherence metrics are **novel contributions**

### 3. ICD-9 Hierarchical Generation Infrastructure ✅

**Implementation:**
- `icd9_hierarchy.py` (315 lines): Category extraction, hierarchy mapping, category co-occurrence
- `hierarchical_tokenizer.py` (242 lines): Dual-vocabulary tokenizer (categories + codes)
- `test_hierarchy.py`, `test_hierarchy_full.py`, `test_hierarchical_tokenizer.py`: All tests passed

**Results (50k patients):**
```
Sparsity reduction: 6,985 codes → 943 categories (7.4x)
Coverage improvement: 3.25% (code) → 26.07% (category) = 8.0x
Avg co-occurrence: 5.7 (code) → 31.9 (category) = 5.6x
```

**Architecture:**
- Token IDs 0-6: Special tokens (<s>, <pad>, </s>, <unk>, <v>, <\v>, <mask>)
- Token IDs 7-949: Category tokens (943 categories)
- Token IDs 950-7934: Code tokens (6,985 codes)
- Total vocab size: 7,935 tokens

---

## Remaining Work

### 4. Hierarchical Dataset (2-3 hours)

**Needed:**
- `hierarchical_dataset.py`: Dataset that outputs category sequences for training
- Modify `EHRDataCollator` to support category-based corruption
- Training format: `<s> <v> cat1 cat2 <\v> <v> cat3 <\v> </s>`

**Challenge:**
- Model needs to learn category-level patterns (better coverage)
- Expansion to codes happens during generation, not training

### 5. Hierarchical Model Integration (4-6 hours)

**Needed:**
- Modify `PromptBartWithDemographicPrediction` to support category vocabulary
- Add category-level LM head (943 categories vs 6,985 codes)
- Maintain compatibility with existing multi-task learning (age/sex)

**Options:**
1. **Category-only training:** Train on category sequences, expand during generation
2. **Dual training:** Train on both categories and codes (more complex)
3. **Progressive training:** Train on categories first, fine-tune on codes

**Recommendation:** Option 1 (simplest, most aligned with hierarchy benefits)

### 6. Two-Stage Generation (2-3 hours)

**Needed:**
- Modify `generate.py` to support hierarchical generation
- Stage 1: Generate category sequence using model
- Stage 2: Expand each category to 1-3 specific codes using sampling

**Pseudo-code:**
```python
# Stage 1: Generate categories
category_sequence = model.generate(
    input_ids=prompt,
    max_length=max_categories,
    vocab_mask=category_token_mask  # Only allow category tokens
)

# Stage 2: Expand to codes
code_sequence = []
for category in category_sequence:
    candidate_codes = hierarchy.get_category_codes(category)
    n_codes = sample_from_distribution(min=1, max=3, mode=2)
    sampled_codes = random.sample(candidate_codes, n_codes)
    code_sequence.extend(sampled_codes)
```

### 7. Comprehensive Evaluation (1-2 hours)

**Needed:**
- Run `evaluate_semantic_coherence.py` with new model
- Run `evaluate_medical_validity.py` to ensure no regression
- Compare metrics before/after hierarchical generation

**Target Metrics:**
- JS divergence: <0.3 (current: 0.61)
- Co-occurrence score: >20 (current: 2.80)
- Top-100 overlap: >0.5 (current: 0.04)
- Medical validity: maintain 99% age, 96% sex

---

## Git Status

**Branch:** `feature/cooccurrence-learning`

**Commits:**
1. `f460f4c`: Add co-occurrence regularization loss to improve semantic coherence
2. `0930114`: Add ICD-9 hierarchical generation infrastructure

**Files Changed:**
- Modified: `config.py`, `trainer.py`
- New: `cooccurrence_utils.py`, `test_cooccurrence.py`
- New: `icd9_hierarchy.py`, `hierarchical_tokenizer.py`
- New: `test_hierarchy.py`, `test_hierarchy_full.py`, `test_hierarchical_tokenizer.py`
- New: `docs/analysis/`, `docs/reference/`, `docs/diagrams/`

**Ready for merge:** No (work in progress)

---

## Next Steps

**Immediate (1-2 hours):**
1. Implement `hierarchical_dataset.py` with category-based training
2. Test dataset with small sample

**Short-term (4-6 hours):**
3. Integrate hierarchical tokenizer into model
4. Train model on category sequences
5. Implement two-stage generation

**Final (1-2 hours):**
6. Run comprehensive evaluation
7. Compare metrics before/after
8. Document results
9. Create pull request

**Estimated total remaining time:** 6-9 hours

---

## Technical Decisions

### Co-occurrence Loss Weight (0.05)
- Higher than age/sex (0.005) because semantic coherence is priority
- Lower than LM loss (1.0) to avoid overfitting to training co-occurrences
- Can tune during evaluation

### Hierarchical Generation Strategy
- **Chosen:** Category-only training + expansion during generation
- **Rationale:**
  - Simplest implementation
  - Maximum benefit from 8x coverage improvement
  - Maintains code-level granularity where needed
- **Alternative:** Could fine-tune on codes after category pre-training (future work)

### Category Expansion Strategy
- Sample 1-3 codes per category during generation
- Use frequency distribution from training data
- Could add learned expansion model (future work)

---

## Notes for Continuation

1. **Dataset implementation:** Use `hierarchical_tokenizer.encode_patient_as_categories()` for encoding
2. **Model integration:** May need to resize token embeddings for dual vocabulary
3. **Generation:** Use `no_repeat_ngram_size=1` at both stages to prevent duplicates
4. **Evaluation:** Expect improvements in co-occurrence, possible slight drop in perplexity (acceptable tradeoff)

---

## Contact

**Author:** jalengg <jalen.jiang2@gmail.com>
**Branch:** feature/cooccurrence-learning
**Status:** Infrastructure complete, integration pending
