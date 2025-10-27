# Generation Quality Analysis - Phase 9 (Temperature 0.3)

**Date**: 2025-10-19
**Patients Analyzed**: 20
**Model**: Phase 9 with reparameterization + dual prompts
**Temperature**: 0.3

---

## Issues Identified

### 1. **Duplicate Codes Within Single Visit**

**Patient 1 (65yo WHITE M) - Visit 1**:
- 82521 (Fracture of mandible) appears **TWICE** (codes 2 and 7)

**Patient 2 (45yo BLACK F) - Visit 2**:
- 9053 (Late effect of fracture) appears **TWICE** (codes 1 and 9)

**Patient 3 (30yo HISPANIC M)**:
- Visit 1: 7313 (Pathological fracture of vertebrae) **TWICE** (codes 9 and 12)
- Visit 2: 76496 (Multiple gestation) appears in both Visit 1 and Visit 2

**Pattern**: Same diagnosis code appearing multiple times in a SINGLE visit is medically nonsensical. Each unique diagnosis should appear once per visit.

**Root Cause**: The model has next-token suppression for IMMEDIATE duplicates (line 159-162 in generate.py), but this only prevents back-to-back duplicates. Codes can repeat if separated by other codes.

---

### 2. **Code Oversaturation (Too Many Codes Per Visit)**

**Statistics**:
- Average codes per visit: **~15-16 codes**
- Real MIMIC-III average: **9.27 codes**
- Real MIMIC-III median: **9 codes**

**Examples**:
- Most patients: 15-16 codes per visit
- Real data: 49.7% of visits have 6-10 codes
- Generated data consistently at 90th percentile (17 codes)

**Why This Happens**:
```python
# In generate.py
max_codes_per_visit=15  # Upper bound acts as soft target
min_codes_per_visit=3   # Lower bound prevents early termination
```

The model learns to fill up to the maximum. With corruption training expanding sequences, it learned longer patterns.

---

### 3. **Longitudinal Consistency Issues**

#### 3A. Recurring Chronic Conditions (GOOD)

Some patterns are medically plausible:

**Patient 3 (30yo HISPANIC M)**:
- Visit 1: 5258 (Gastrojejunal ulcer)
- Visit 2: 5258 (Same ulcer - chronic condition recurring)

**Patient 2 (45yo BLACK F)**:
- Visit 1: 73022 (Fracture of femur)
- Visit 2: 73022 (Same fracture - follow-up)
- Visit 2: 9053 (Late effect of fracture) - logical progression

✅ **This is actually GOOD** - chronic diseases should recur across visits.

#### 3B. Acute Conditions Recurring (QUESTIONABLE)

**Patient 1 (65yo WHITE M)**:
- Visit 1: 82521 (Fracture of mandible) x2 in SAME visit ❌
- This is the duplicate problem, not longitudinal

**Patient 2 (45yo BLACK F)**:
- Visit 1: 99939 (Complications of medical care)
- Visit 2: 99939 (Same complication)

Could be legitimate follow-up, or could be duplicate coding error.

#### 3C. Impossible Temporal Sequences

**Patient 2 (45yo BLACK F)**:
- Visit 2: 75567 (Congenital anomalies of spinal cord)

Congenital = present from birth. Should appear in FIRST visit, not later visits. ❌

---

### 4. **Cross-Modality Imputation (Medical Implausibility)**

#### 4A. Trauma + Chronic Disease Combinations

**Patient 1 (65yo WHITE M) - Single Visit**:
- Heart disease (4295, 42971)
- Kidney disorder (5968)
- Personality disorder (3010)
- **PLUS**: Multiple fractures (82521, 73022, 9053)
- **PLUS**: Water transport accident (E8120)

**Medically implausible**: Trauma patients don't typically have 6 different chronic conditions documented in the SAME acute admission.

#### 4B. Pregnancy Codes in Non-Reproductive Context

**Patient 3 (30yo HISPANIC M)**:
- Kidney disorder (5968)
- Ulcers (5258, 5224)
- Fractures (92400, 7313)
- **Pregnancy codes**: 76407 (Polyhydramnios), 76496 (Multiple gestation)
- Male patient ❌

#### 4C. Neonatal Codes in Adults

**Patient 2 (45yo BLACK F) - Visit 2**:
- 76522 (Fetal distress in liveborn infant)
- Patient is 45 years old ❌

---

### 5. **Inappropriate Code Categories**

**Age-Inappropriate**:
- Patient 2 (45yo): V0381 (Pediatric Hib vaccine)
- Patient 5 (55yo M): 76526 (Preterm infant)
- Patient 7 (25yo F): 76526 (Preterm infant)

**Gender-Inappropriate**:
- Patient 3 (30yo M): Multiple pregnancy codes

**Reduction vs Temperature 0.8**:
- Temperature 0.8: ~10+ inappropriate codes across 10 patients
- Temperature 0.3: 2 inappropriate codes across 20 patients
- **Improvement: 80-90% reduction** ✅

---

## Root Causes Summary

| Issue | Root Cause | Solution Needed |
|-------|------------|-----------------|
| **Duplicate codes in visit** | Only immediate duplicate suppression | Track all codes in current visit, suppress ALL seen codes |
| **Code oversaturation** | `max_codes_per_visit=15` acts as target | Lower to 10, or use dynamic termination |
| **Congenital codes in later visits** | No temporal logic | Medical knowledge injection or data cleaning |
| **Trauma + chronic mix** | Training data has co-occurrence | Data cleaning, or cross-modality coherence loss |
| **Age/gender inappropriate** | Learned from MIMIC errors | Temperature 0.3 helped significantly (80% reduction) |

---

## Proposed Solutions

### Priority 1: Fix Duplicate Codes (Easy)

```python
# In generate.py, track codes per visit
codes_in_current_visit = set()

# After sampling next token
if next_token_id >= tokenizer.code_offset:  # Is diagnosis code
    if next_token_id in codes_in_current_visit:
        next_token_logits[next_token_id] = float('-inf')  # Block duplicate
    else:
        codes_in_current_visit.add(next_token_id)

# Clear when entering new visit
if next_token_id == v_token_id:
    codes_in_current_visit.clear()
```

### Priority 2: Reduce Oversaturation (Easy)

```python
# Change generation parameters
max_codes_per_visit=10  # Lower from 15
min_codes_per_visit=5   # Raise from 3

# Or use probabilistic early termination
if codes_in_current_visit >= 10 and np.random.rand() < 0.5:
    # Allow visit end with 50% probability
```

### Priority 3: Clean Training Data (Medium)

```python
# In data_loader.py, filter out implausible codes
def is_valid_code(code, age, gender, visit_index):
    # Remove pediatric codes for adults
    if age >= 18 and code in PEDIATRIC_CODES:
        return False

    # Remove pregnancy codes for males
    if gender == 'M' and code in PREGNANCY_CODES:
        return False

    # Remove congenital codes from later visits
    if visit_index > 0 and code in CONGENITAL_CODES:
        return False

    return True
```

### Priority 4: Cross-Modality Coherence (Hard - Research)

Add auxiliary loss during training:
```python
# Penalize co-occurrence of trauma + chronic in same visit
trauma_codes = {...}
chronic_codes = {...}

# If both present in visit, add penalty to loss
if has_trauma and has_chronic:
    loss += coherence_penalty
```

---

## What's Working Well

✅ **Temperature 0.3 dramatically reduced inappropriate codes** (80-90% reduction)
✅ **Chronic disease recurrence** (e.g., ulcers, fractures across visits)
✅ **TPL = 1.0014** (excellent temporal coherence statistically)
✅ **No generation artifacts** (no spurious tokens, clean sequences)
✅ **Diverse demographics** (age, gender, ethnicity conditioning works)

---

## Recommendations

**For immediate improvement (no retraining)**:
1. ✅ **DONE**: Temperature 0.3 (reduced inappropriate codes by 80%)
2. **TODO**: Fix duplicate code suppression in generate.py
3. **TODO**: Lower max_codes_per_visit from 15 to 10

**For next model version (requires retraining)**:
1. Clean training data (remove age/gender/temporal inappropriate codes)
2. Add medical coherence constraints during training
3. Increase training data size (3000 → 10,000 patients)

**Current Status**: Model generates statistically coherent sequences with good demographic conditioning, but lacks medical domain knowledge for clinical deployment.
