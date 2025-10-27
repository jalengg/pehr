# Semantic Coherence Fix - Training Run

## Problem Diagnosis

Current model (aux weights: age=0.01, sex=0.2) achieves:
- **Medical Validity: PASS**
  - 99% age-appropriate codes
  - 96% sex-appropriate codes
  - 0% duplicate codes

- **Semantic Coherence: FAIL**
  - JS Divergence: 0.606 (target: <0.3) - code frequencies drastically different
  - Top-100 Overlap: 0.038 (target: >0.5) - only 4% of common codes learned
  - Co-occurrence Score: 3.39 (target: >20) - codes rarely appear together in training

**Root Cause**: Auxiliary age/sex classification losses (especially sex_loss_weight=0.2) dominate training, causing the model to prioritize generating medically valid codes over learning realistic code frequency distributions and co-occurrence patterns.

## Hypothesis

Reducing auxiliary loss weights from 0.01/0.2 to 0.001/0.001 will shift model focus back to language modeling (learning code distributions) while maintaining weak validity guidance.

## Configuration Changes

```python
# Previous (aux0.01-0.2)
age_loss_weight: 0.01
sex_loss_weight: 0.2

# New (aux0.001-0.001)
age_loss_weight: 0.001  # 10x reduction
sex_loss_weight: 0.001  # 200x reduction
```

Training for 30 epochs (loss plateaus around epoch 30).

## Expected Outcomes

**Best Case:**
- Maintain medical validity >95% (slight degradation acceptable)
- Improve JS Divergence to <0.3
- Improve Top-100 Overlap to >0.5
- Improve Co-occurrence Score to >20

**Acceptable Trade-off:**
- Medical validity drops to 90-95%
- Semantic coherence improves significantly (JS <0.4, Top-100 >0.3, Co-occur >10)

**Worst Case:**
- Medical validity drops below 85%
- Solution: Try intermediate weights (0.005/0.005) or curriculum learning

## Evaluation Plan

After training completes:
1. Run `evaluate_medical_validity.py` to assess medical validity
2. Run `evaluate_semantic_coherence.py` to assess semantic coherence
3. Compare against baseline (aux0.01-0.2) backed up in `best_model_aux0.01-0.2.pt`

## Checkpoint Backup

Current best model saved to: `/scratch/jalenj4/promptehr_checkpoints/best_model_aux0.01-0.2.pt`
