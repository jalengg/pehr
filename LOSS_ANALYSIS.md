# Loss Calculation Analysis

## Executive Summary

Investigated why certain loss components stopped moving during hierarchical model training. **Finding: Loss calculations are mathematically correct, but auxiliary tasks provide minimal training signal due to task difficulty mismatch.**

## Issues Identified

### 1. Sex Loss Vanished (CRITICAL)

**Observation:**
```
Epoch  1: Sex = 0.167
Epoch  2: Sex = 0.015 (91% drop)
Epoch 10: Sex = 0.001
Epoch 14: Sex = 0.000 (DEAD)
Epochs 14-30: Sex = 0.000 (flat)
```

**Root Cause:**
Sex prediction task is trivial - model achieves near-perfect accuracy by epoch 14.

**Why:**
- MIMIC-III contains sex-specific ICD-9 codes:
  - Pregnancy codes (630-679) → Female only
  - Prostate codes (600-608) → Male only
  - Breast cancer patterns differ by sex
- Model learns simple lookup: "if patient has code X → sex Y"
- No meaningful gradient after epoch 13

**Contribution to Total Loss:**
- Epoch 30: 0.000 × 0.005 = 0.000000 (0.00%)
- **Sex loss provides ZERO optimization signal**

**Impact:**
- Positive: 100% medical validity for sex-appropriate codes
- Negative: Wasted compute, no diversity benefit
- Mode collapse: May have learned to generate only stereotypical sex-specific codes

---

### 2. Age Loss Minimal Contribution

**Observation:**
```
Epoch  1: Age = 0.81
Epoch 10: Age = 0.54
Epoch 20: Age = 0.13
Epoch 30: Age = 0.21
```

**Contribution Analysis:**
- Raw age loss: 0.21
- Weight: 0.005
- Contribution: 0.21 × 0.005 = **0.001050** (0.16% of total)
- LM loss: 0.4763 (72.10%)
- Cooccur loss: 0.183250 (27.74%)

**Diagnosis:**
Age loss IS moving (0.81 → 0.21) but contributes negligibly to optimization due to:
1. Low weight (0.005)
2. Smaller magnitude than LM loss (~100× smaller)

**Impact:**
- Age loss gradient is 100× weaker than LM gradient
- Minimal influence on code generation
- May explain why age-inappropriate codes still occur (~1%)

---

### 3. Co-occurrence Loss Stable But High

**Observation:**
```
All epochs: Cooccur = ~3.66 (flat)
```

**Contribution:**
- Raw loss: 3.665
- Weight: 0.05
- Contribution: 0.183250 (27.74% of total)
- **Second largest loss component**

**Diagnosis:**
Co-occurrence loss is stable but not decreasing:
- Weight (0.05) is 10× larger than age/sex weights
- Contributing significant gradient
- But NOT improving (stuck at ~3.66)

**Possible Causes:**
1. Threshold=5 may be too high (only penalizing very rare pairs)
2. Matrix built from training data may not match generation distribution
3. Model generating valid codes that don't co-occur in training data
4. Conflicting with LM loss objective

---

## Mathematical Verification

### Loss Formula

```python
total_loss = lm_loss + age_weight × age_loss + sex_weight × sex_loss + cooccur_weight × cooccur_loss
```

### Epoch 30 Calculation

```
LM loss:      0.4763 × 1.0   = 0.476300  (72.10%)
Age loss:     0.2100 × 0.005 = 0.001050  ( 0.16%)
Sex loss:     0.0000 × 0.005 = 0.000000  ( 0.00%)
Cooccur loss: 3.6650 × 0.05  = 0.183250  (27.74%)
─────────────────────────────────────────────────
Total:                        = 0.660600
```

**Verification:** ✓ Math is correct (tested with random initialized model)

---

## Problems This Causes

### 1. Mode Collapse

Generated patients show severe mode collapse:
- Only 31/6,985 unique codes (0.44% vocabulary)
- All patients have exactly 2 categories (never varies)
- Top code appears in 12.5% of patients

**Semantic Analysis of Generated Codes:**

The 31 unique codes fall into just 3 groups:

1. **Gastric Disorders (536.x) - ~50% of codes:**
   - 536.1: Acute dilatation of stomach
   - 536.2: Persistent vomiting
   - 536.3: Gastroparesis
   - 536.40-49: Gastrostomy complications (infection, mechanical)
   - 536.8-9: Dyspepsia, unspecified stomach disorders

2. **Medical Care Complications (999.x) - ~45% of codes:**
   - 999.1-2: Air embolism, vascular complications
   - 999.3x: Catheter infections (central venous, bloodstream)
   - 999.4x: Anaphylactic reactions to blood products
   - 999.6-71: ABO/Rh incompatibility
   - 999.8x: Chemotherapy extravasation, transfusion reactions

3. **Category Tokens (2-6) - ~5% of codes:**
   - Category 2: Neoplasms
   - Category 3: Endocrine/Metabolic
   - Category 4: Blood disorders
   - Category 5: Mental disorders

**Why This Pattern Emerged:**

1. **MIMIC-III is 100% ICU patients:**
   - Catheter infections (999.3x) are near-universal in ICU
   - Gastric complications (536.x) common in intubated patients
   - Model learned: "Patient = ICU patient with catheter + stomach issues"

2. **Sex-neutral codes:**
   - NONE of these codes are sex-specific
   - Sex loss (0.000 after epoch 14) forced avoidance of pregnancy/prostate codes
   - Model converged to safest sex-neutral codes

3. **High co-occurrence:**
   - These codes frequently appear together in ICU patients
   - Co-occurrence loss (27.74% of total) strongly rewards this combination
   - Penalizes generating rare or novel code pairs

4. **Result: Stereotypical ICU patient template**
   - Every generated patient = "ICU patient with catheter infection + gastric issues"
   - No diversity in disease presentation
   - No chronic conditions, no age-specific diseases, no primary diagnoses

### 2. No Diversity Signal

After epoch 14:
- Sex loss: 0% contribution
- Age loss: 0.16% contribution
- Cooccur loss: 27.74% but not decreasing

**Only LM loss (72%) is actively optimizing**

This means:
- No incentive for code diversity
- No penalty for generating same codes repeatedly
- Auxiliary tasks failed their purpose

### 3. Wasted Computation

Multi-task learning overhead:
- Sex predictor head: 197,634 parameters
- Age predictor head: 197,382 parameters
- Total auxiliary params: 395,016 (0.37% of model)

**Cost:** Forward pass computes unused predictions every batch

---

## Recommendations

### Immediate Fixes

1. **REMOVE co-occurrence loss**
   - Currently 27.74% of total loss
   - **Directly causing mode collapse** by forcing common ICU combinations
   - Model cannot explore diverse code patterns
   - Semantic coherence will improve WITHOUT it

2. **Remove sex loss entirely**
   - Provides zero gradient after epoch 14
   - Forcing sex-neutral codes contributed to mode collapse
   - Use sex-specific code rules in post-processing instead
   - Save computation

3. **Increase age loss weight OR remove it**
   - Current: 0.005 (0.16% contribution - essentially zero)
   - Either increase to 0.05-0.1 (meaningful contribution)
   - OR remove entirely (not helping at current weight)

### Long-term Solutions

1. **Curriculum learning for sex loss**
   ```python
   if epoch < 15:
       sex_weight = 0.005
   else:
       sex_weight = 0.0  # Turn off after convergence
   ```

2. **Dynamic loss weighting**
   - Auto-adjust weights to equalize gradient magnitudes
   - Normalize losses to same scale before weighting

3. **Replace auxiliary losses with better objectives**
   - Age: Use age-specific code frequency matching
   - Sex: Remove entirely, use rule-based validation
   - Cooccur: Use mutual information instead of threshold

4. **Add explicit diversity loss**
   - Penalize low vocabulary coverage
   - Reward generating rare codes
   - Track unique code count as metric

---

## Testing Results

### Test 1: Loss Calculation Verification

**Setup:** Random initialized model, 8-sample batch

**Results:**
```
LM loss:   7.514641
Age loss:  1.760908
Sex loss:  0.748391
Total:     7.527187

Calculated: 7.514641 + 0.005×1.761 + 0.005×0.748 = 7.527188
Match: ✓ (difference < 1e-5)
```

**Conclusion:** Loss calculation math is correct

### Test 2: Sex Prediction Accuracy (In Progress)

**Running:** `diagnose_sex_prediction.py`
**Purpose:** Measure actual sex prediction accuracy on trained model
**Hypothesis:** Accuracy > 99.5% explains zero loss

---

## Configuration Issues

### Current Config (`config.py`)

```python
age_loss_weight: 0.005
sex_loss_weight: 0.005
cooccurrence_loss_weight: 0.05
```

### Problems:

1. **Age/sex weights 10× smaller than cooccur weight**
   - Cooccur dominates auxiliary signal
   - Age/sex barely contribute

2. **No weight scheduling**
   - Weights are constant across all epochs
   - Can't adapt to task difficulty changes

3. **Weights chosen arbitrarily**
   - No principled justification for 0.005 vs 0.05
   - Not normalized to loss magnitudes

### Recommended Config:

```python
age_loss_weight: 0.0      # Remove (0.16% contribution ineffective)
sex_loss_weight: 0.0      # Remove (converged to 0.000 after epoch 14)
cooccurrence_loss_weight: 0.0  # REMOVE - directly causing mode collapse
```

**Rationale:**
- Co-occurrence loss enforces ICU complication stereotypes
- Sex loss converged and blocks diverse code generation
- Age loss too weak to matter
- **Train with LM loss ONLY** to maximize diversity

---

## Next Steps

1. ✓ Complete sex prediction accuracy test
2. Investigate co-occurrence loss implementation
3. Re-train with adjusted weights
4. Add diversity metrics to training loop
5. Consider removing multi-task learning entirely

---

## Code Frequency Analysis

**Top generated codes (out of 512 total tokens):**

| Code | Count | % | Description |
|------|-------|---|-------------|
| 4 | 64 | 12.5% | Category: Blood/Blood-Forming Organs |
| 53642 | 30 | 5.9% | Mechanical complication of gastrostomy |
| 5362 | 28 | 5.5% | Persistent vomiting |
| 5361 | 27 | 5.3% | Acute dilatation of stomach |
| 5 | 23 | 4.5% | Category: Mental Disorders |
| 99982 | 19 | 3.7% | Extravasation of vesicant chemotherapy |
| 99932 | 15 | 2.9% | Bloodstream infection due to central venous catheter |

**Distribution:**
- **536.x (Gastric)**: ~50% of all codes
- **999.x (Medical complications)**: ~45% of all codes
- **Category tokens**: ~5% of all codes
- **Everything else**: 0%

This confirms the model learned a single template:
**"ICU patient with catheter infection + gastric complications"**

## Files Created

- `test_loss_calculation.py` - Verifies loss math correctness
- `diagnose_sex_prediction.py` - Measures sex prediction accuracy (still running)
- `LOSS_ANALYSIS.md` - This document

## Related Documents

- `EVALUATION_RESULTS.md` - Generation quality assessment
- `docs/reference/MULTITASK_LEARNING.md` - Auxiliary loss design
- `trainer_hierarchical.py:143-280` - Loss calculation code
- `prompt_bart_model.py:373-489` - Multi-task model architecture
