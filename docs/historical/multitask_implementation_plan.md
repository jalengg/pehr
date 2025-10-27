# Multi-Task Learning Implementation Plan

**Date:** October 23, 2025
**Goal:** Implement auxiliary age/sex prediction + fix duplicate generation
**Time:** ~6-8 hours implementation + 4 hours retraining

---

## Quick Reference

### Changes Summary

| Component | Current | Target |
|-----------|---------|--------|
| Demographics | age, sex, race | age, sex |
| x_cat shape | [batch, 2] | [batch, 1] |
| cat_cardinalities | [2, 6] | [2] |
| Model | PromptBartModel | + age/sex predictors |
| Loss | LM only | LM + 0.3×age + 0.2×sex |
| Generation | Manual loop | model.generate() |
| Duplicate prevention | None | no_repeat_ngram_size=1 |

### Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Jaccard | 0.403 | 0.42-0.45 |
| Age violations | ~30% | <2% |
| Duplicates | High | ~0% |

---

## Implementation Steps

See **MULTITASK_LEARNING.md** for full technical details and code.
See **MEDICAL_VALIDITY.md** for validation metrics and constraints.

### Phase 1: Remove Race (1 hour)
1. Update config.py: `cat_cardinalities: [2]` (not [2,6])
2. Update data_loader.py: Remove ethnicity_id from PatientRecord
3. Update dataset.py: x_cat shape [batch, 1]
4. Update tests: Fix x_cat assertions

### Phase 2: Add Auxiliary Heads (2 hours)
1. Create PromptBartWithDemographicPrediction class in prompt_bart_model.py
2. Add age_predictor (regression) and sex_predictor (classification)
3. Implement combined loss: `total = lm + 0.3×age + 0.2×sex`
4. Update trainer.py to log separate losses

### Phase 3: Fix Generation (2 hours)
1. Rewrite generate_patient_sequence_conditional() in generate.py
2. Use model.generate() instead of manual loop
3. Add no_repeat_ngram_size=1 parameter
4. Pass x_num/x_cat through generation

### Phase 4: Retrain (4 hours GPU)
1. Update train.slurm (48h time, 64GB RAM)
2. Submit job: `sbatch train.slurm`
3. Monitor losses (age/sex should decrease)

### Phase 5: Evaluate (1 hour)
1. Run generate_from_test.py
2. Create evaluate_medical_validity.py
3. Measure age/sex violation rates

---

## Success Criteria

- ✓ Age loss < 5.0 (predicting age within ~2 years)
- ✓ Sex loss < 0.05 (>95% accuracy)
- ✓ Jaccard ≥ 0.42
- ✓ Age violations < 2%
- ✓ Sex violations < 1%
- ✓ Duplicate codes ~0%

---

For complete implementation code and literature references, see **MULTITASK_LEARNING.md**.
