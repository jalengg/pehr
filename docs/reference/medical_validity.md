# Medical Validity Constraints for EHR Generation

**Date:** October 23, 2025
**Purpose:** Document medically impossible/implausible code patterns and constraints for validation

---

## Table of Contents

1. [Overview](#overview)
2. [Age-Inappropriate Codes](#age-inappropriate-codes)
3. [Sex-Inappropriate Codes](#sex-inappropriate-codes)
4. [Race Removal Justification](#race-removal-justification)
5. [Validation Metrics](#validation-metrics)

---

## Overview

### Current Medical Validity Issues (10k/50epoch Model)

**Analysis Date:** October 23, 2025
**Test Set:** 20 random samples from 2000 held-out patients
**Finding:** ~30% of generated codes are medically inappropriate for patient demographics

**Examples from RECONSTRUCTION_ANALYSIS.md:**

| Patient ID | Age | Sex | Target Codes | Generated Codes | Issues |
|-----------|-----|-----|--------------|-----------------|--------|
| 1 | 0 | F | V3000, V053, V290 | V3000, **3572**, **42979** | Diabetic neuropathy + atrial fib in newborn |
| 2 | 56 | M | 9982, V420, 42789, 2859, 4019 | 9982, V420, 42789, 2859, 4019, **42979×4** | Atrial fib duplicated 4 times |
| 3 | 0 | F | V3101, 769, 7461, 7454, 76525, V290 | V3101, 7461, 7454, 76525, V290, **34690**, **42979** | Migraine + atrial fib in newborn |

**Pattern:** Model frequently generates **42979 (atrial fibrillation)** regardless of age, suggesting frequency bias from training data.

---

## Age-Inappropriate Codes

### Neonatal-Only Codes (Age 0-1 years)

These codes should **NEVER** appear for patients age ≥ 2:

| ICD-9 Code | Description | Valid Age Range |
|-----------|-------------|-----------------|
| **V30** family | Liveborn infants | 0 only |
| V3000 | Single liveborn, born in hospital | 0 only |
| V3001 | Single liveborn, born before admission | 0 only |
| V3101 | Twin birth, mate liveborn, born in hospital | 0 only |
| **76** family | Fetal and neonatal conditions | 0-1 |
| 769 | Respiratory distress syndrome in newborn | 0 only |
| 7461 | Transitory tachypnea of newborn | 0 only |
| 7454 | Abnormal weight loss (neonatal) | 0-1 |
| 76515 | Extreme immaturity, 1000-1249 grams | 0 only |
| 76525 | Other preterm infants, 1000-1249 grams | 0 only |
| 7742 | Other preterm infants | 0 only |
| **V05** family | Prophylactic vaccination (newborn) | 0-1 |
| V053 | Prophylactic vaccination - viral hepatitis | 0-2 (newborn dose) |
| V502 | Routine circumcision | 0 only |
| **V29** family | Observation of newborn | 0 only |
| V290 | Observation for suspected infectious condition | 0-1 |

**Rule:** If patient age > 1, **EXCLUDE** all codes starting with V30*, V31*, 76*, 77*, V502, V290.

---

### Adult/Geriatric Codes (Minimum Age Requirements)

These codes should **RARELY or NEVER** appear before specified ages:

| ICD-9 Code | Description | Minimum Age | Reasoning |
|-----------|-------------|-------------|-----------|
| **42979** | **Atrial fibrillation** | **18** | Extremely rare in children, prevalence increases with age |
| 41401 | Coronary atherosclerosis | 30 | Requires decades of plaque buildup |
| 412 | Old myocardial infarction | 30 | Rare MI before age 30 |
| 4139 | Chronic ischemic heart disease | 30 | Long-term cardiac condition |
| 4240 | Mitral valve disorders | 40 | Usually from rheumatic fever or degenerative changes |
| 3572 | Polyneuropathy in diabetes | 10 | Requires years of diabetes (Type 2 rare in children) |
| 2449 | Hypothyroidism | 10 | Congenital hypothyroidism coded differently |
| 2720 | Pure hypercholesterolemia | 18 | Adult metabolic disorder |
| 2749 | Disorder of lipoid metabolism | 18 | Adult metabolic disorder |
| 5855 | Chronic kidney disease, Stage V | 18 | Long-term kidney decline |
| 34690 | Migraine | 10 | Can occur in children but very rare in infants |

**Rule:** Probabilistic - these codes CAN appear below minimum age but should be heavily penalized.

---

### Pediatric Constraints (Age < 18)

Codes that are **uncommon** in children but not impossible:

| ICD-9 Code | Description | Pediatric Prevalence | Adult Prevalence |
|-----------|-------------|---------------------|------------------|
| 4019 | Hypertension | <1% | ~30% |
| 250.00 | Diabetes mellitus Type II | <0.1% | ~10% |
| 42731 | Atrial fibrillation | <0.01% | ~5% |
| 5855 | CKD Stage V | <0.1% | ~2% |

**Rule:** Don't hard-exclude, but model should learn these are low-probability for pediatric patients.

---

### Current Model Failures - Age Examples

**Example 1: Newborn with Adult Cardiac Conditions**
```
Patient: Age 0, Female, White
Target:    [V3000, V053, V290]
Generated: [V3000, 3572, 42979]

Medical Validity Check:
✓ V3000 (Single liveborn) - VALID for age 0
✗ 3572 (Polyneuropathy in diabetes) - INVALID (requires min age ~10)
✗ 42979 (Atrial fibrillation) - INVALID (requires min age ~18)

Severity: CRITICAL - Two medically impossible codes
```

**Example 2: Premature Infant with Migraine**
```
Patient: Age 0, Female, White
Target:    [V3101, 769, 7461, 7454, 76525, V290]
Generated: [V3101, 7461, 7454, 76525, V290, 34690, 42979]

Medical Validity Check:
✓ V3101, 7461, 7454, 76525, V290 - VALID for age 0
✗ 34690 (Migraine) - IMPLAUSIBLE (min age ~10)
✗ 42979 (Atrial fibrillation) - INVALID (min age ~18)

Severity: CRITICAL - Two medically inappropriate codes
```

---

## Sex-Inappropriate Codes

### Female-Only Codes

**Pregnancy and Childbirth (630-679):**
- Should **NEVER** appear for male patients
- Examples: V27* (Live birth outcomes), 650* (Normal delivery), 674* (Complications of pregnancy)

**Gynecological Conditions:**
- 174* (Breast cancer - predominantly female, but can occur in males at ~1%)
- 180-184 (Female genital organ malignancies)
- 218* (Uterine leiomyoma)
- 625* (Pain and other symptoms associated with female genital organs)

**Rule:** If sex=M, **EXCLUDE** all codes 630-679, 218*, 625*, and most 174*.

---

### Male-Only Codes

**Prostate Conditions:**
| ICD-9 Code | Description |
|-----------|-------------|
| 600* | Hyperplasia of prostate |
| 185 | Malignant neoplasm of prostate |
| 601* | Inflammatory diseases of prostate |

**Testicular Conditions:**
| ICD-9 Code | Description |
|-----------|-------------|
| 186* | Malignant neoplasm of testis |
| 608* | Other disorders of male genital organs |

**Rule:** If sex=F, **EXCLUDE** all codes 600*, 185, 601*, 186*, 608*.

---

### Current Model Failures - Sex Examples

**Note:** Sex-inappropriate codes not observed in current 20-sample evaluation, but likely present in larger samples due to lack of sex constraint.

**Expected failure mode:**
```
Patient: Age 65, Male
Target:    [4019, 42979, 5855]
Generated: [4019, 42979, 5855, 650]  ← 650 (Normal delivery) added for male!

Medical Validity Check:
✓ 4019, 42979, 5855 - VALID for age 65
✗ 650 (Normal delivery) - IMPOSSIBLE for male patient

Severity: CRITICAL - Biologically impossible
```

---

## Race Removal Justification

### Medical Perspective

**Race is NOT a biological determinant:**
- Race is a **social construct**, not a genetic category
- Most diseases have **zero race-specific** coding in ICD-9
- Genetic conditions (sickle cell, Tay-Sachs) are **ancestry-specific**, not race-specific

**ICD-9 does NOT have race-specific codes:**
- No codes like "42979-W" (atrial fib in White patients)
- Diagnosis is based on physiology, not demographics
- Using race as conditioning variable adds noise without medical signal

### Statistical Perspective

**MIMIC-III Ethnicity Data Quality Issues:**

1. **High missingness:** ~15% of admissions have NULL ethnicity
2. **Inconsistent coding:** Same patient coded as "WHITE", "WHITE - RUSSIAN", "WHITE - OTHER EUROPEAN"
3. **Broad categories:** "ASIAN" combines East Asian, South Asian, Southeast Asian (very different genetic backgrounds)
4. **No medical relevance:** ICD-9 codes don't correlate strongly with MIMIC ethnicity categories

**Evidence from our data (dataset.py:234):**
```python
x_cat: [2] array with [gender_id, ethnicity_id]
cat_cardinalities: [2, 6]  # 2 genders, 6 ethnicity categories
```

**Problem:** 6 ethnicity categories are too coarse to capture genetic diversity, and ICD-9 codes don't have ethnicity-specific patterns.

### Ethical Perspective

**Bias concerns:**
- Training on race-conditioned generation can perpetuate healthcare disparities
- Model might learn biased patterns from historical data (e.g., different treatment for same condition)
- Synthetic data should focus on medical need, not demographic characteristics

**Example of potential bias:**
```
# Model learns from biased historical data:
Patient: Race=Black → Less likely to receive cardiac interventions (historical bias)
Target:  [42979, 4019]  (diagnosis codes)
Missing: [3695] (cardiac catheterization procedure) ← Underrepresented in training

# Model perpetuates bias in synthetic data:
Generated for Black patients: [42979, 4019]  ← Diagnosis only
Generated for White patients: [42979, 4019, 3695]  ← Diagnosis + intervention
```

### Decision: Remove Race from Model

**New demographics:**
- **x_num:** [1] - Age only (continuous)
- **x_cat:** [1] - Sex only (binary: M/F)

**Benefits:**
1. Simpler model (fewer parameters in categorical prompt encoder)
2. No spurious race-based patterns
3. Focuses on medically relevant features (age, sex)
4. Reduces potential bias in synthetic data

**Changes required:**
- Update `cat_cardinalities` from [2, 6] to [2]
- Remove ethnicity from PatientRecord class
- Update data loader to not encode ethnicity

---

## Validation Metrics

### Age-Appropriateness Rate

```python
def age_appropriateness_rate(generated_codes: List[str], age: int) -> float:
    """Calculate percentage of age-appropriate codes.

    Args:
        generated_codes: List of ICD-9 diagnosis codes
        age: Patient age in years

    Returns:
        Fraction of codes that are age-appropriate (0.0 to 1.0)
    """
    appropriate_count = 0

    for code in generated_codes:
        if is_code_valid_for_age(code, age):
            appropriate_count += 1

    return appropriate_count / len(generated_codes) if generated_codes else 0.0


def is_code_valid_for_age(code: str, age: int) -> bool:
    """Check if code is medically valid for given age.

    Returns:
        True if code is appropriate, False if medically impossible/implausible
    """
    # Neonatal-only codes
    neonatal_only = ['V3000', 'V3001', 'V3101', '769', '7461', '7454',
                     '76515', '76525', 'V502', 'V290']
    if code in neonatal_only:
        return age <= 1

    # Adult/geriatric codes with minimum ages
    min_ages = {
        '42979': 18,  # Atrial fibrillation
        '41401': 30,  # Coronary atherosclerosis
        '412': 30,    # Old MI
        '3572': 10,   # Diabetic neuropathy
        '34690': 10,  # Migraine
        '2720': 18,   # Hypercholesterolemia
        '5855': 18,   # CKD Stage V
    }

    if code in min_ages:
        return age >= min_ages[code]

    # Default: code is valid
    return True
```

### Sex-Appropriateness Rate

```python
def sex_appropriateness_rate(generated_codes: List[str], sex: int) -> float:
    """Calculate percentage of sex-appropriate codes.

    Args:
        generated_codes: List of ICD-9 diagnosis codes
        sex: Patient sex (0=Female, 1=Male)

    Returns:
        Fraction of codes that are sex-appropriate (0.0 to 1.0)
    """
    appropriate_count = 0

    for code in generated_codes:
        if is_code_valid_for_sex(code, sex):
            appropriate_count += 1

    return appropriate_count / len(generated_codes) if generated_codes else 0.0


def is_code_valid_for_sex(code: str, sex: int) -> bool:
    """Check if code is medically valid for given sex.

    Args:
        sex: 0=Female, 1=Male

    Returns:
        True if code is appropriate, False if medically impossible
    """
    # Female-only codes (pregnancy, gynecological)
    female_only_prefixes = ['V27', '650', '651', '652', '653', '654', '655',
                           '656', '657', '658', '659', '660', '661', '662',
                           '663', '664', '665', '666', '667', '668', '669',
                           '670', '671', '672', '673', '674', '675', '676',
                           '677', '678', '679', '218', '625']

    if any(code.startswith(prefix) for prefix in female_only_prefixes):
        return sex == 0  # Must be female

    # Male-only codes (prostate, testicular)
    male_only_prefixes = ['600', '185', '601', '186', '608']

    if any(code.startswith(prefix) for prefix in male_only_prefixes):
        return sex == 1  # Must be male

    # Default: code is valid for both sexes
    return True
```

### Overall Medical Validity Score

```python
def medical_validity_score(patient_record: dict) -> dict:
    """Compute comprehensive medical validity metrics.

    Args:
        patient_record: Dict with 'age', 'sex', 'generated_codes'

    Returns:
        Dict with validity metrics
    """
    age = patient_record['age']
    sex = patient_record['sex']
    codes = patient_record['generated_codes']

    age_appropriate = age_appropriateness_rate(codes, age)
    sex_appropriate = sex_appropriateness_rate(codes, sex)

    # Overall validity (all codes must pass both age and sex checks)
    valid_count = 0
    for code in codes:
        if is_code_valid_for_age(code, age) and is_code_valid_for_sex(code, sex):
            valid_count += 1

    overall_validity = valid_count / len(codes) if codes else 0.0

    return {
        'age_appropriate_rate': age_appropriate,
        'sex_appropriate_rate': sex_appropriate,
        'overall_valid_rate': overall_validity,
        'num_age_violations': int((1 - age_appropriate) * len(codes)),
        'num_sex_violations': int((1 - sex_appropriate) * len(codes)),
        'total_violations': int((1 - overall_validity) * len(codes)),
    }
```

### Target Metrics After Multi-Task Learning

| Metric | Current (10k/50ep) | Target (Multi-Task) |
|--------|-------------------|---------------------|
| Age-appropriate rate | ~70% | **>98%** |
| Sex-appropriate rate | Unknown (~95% est.) | **>99%** |
| Overall validity | ~65% | **>97%** |
| Duplicate codes | High (4x same code) | **~0%** (with no_repeat_ngram_size) |

---

## Evaluation Protocol

### Test Set Evaluation

**Dataset:** 2000 held-out patients (20% of 10k)

**Process:**
1. Generate sequences for all 2000 test patients using conditional reconstruction
2. For each patient, compute:
   - Jaccard similarity (reconstruction quality)
   - Age-appropriateness rate
   - Sex-appropriateness rate
   - Overall medical validity
3. Report aggregate statistics:
   - Mean Jaccard
   - Mean age-appropriate rate
   - Mean sex-appropriate rate
   - % patients with zero violations

**Example output:**
```
=== Medical Validity Evaluation ===
Test Patients: 2000
Temperature: 0.3

Reconstruction Quality:
  Average Jaccard: 0.425
  Exact Match Rate: 0.0%

Medical Validity:
  Age-Appropriate Rate: 98.3%
  Sex-Appropriate Rate: 99.7%
  Overall Valid Rate: 98.1%

Violation Statistics:
  Patients with zero violations: 1847 (92.4%)
  Patients with age violations: 134 (6.7%)
  Patients with sex violations: 19 (0.95%)

Most Common Age Violations:
  42979 (Atrial fib) in age < 18: 23 cases
  3572 (Diabetic neuropathy) in age < 10: 12 cases
  34690 (Migraine) in age < 10: 8 cases

Most Common Sex Violations:
  650 (Normal delivery) in males: 11 cases
  185 (Prostate cancer) in females: 8 cases
```

---

## Summary

**Current Issues:**
1. **Age violations:** ~30% of codes inappropriate (atrial fib in newborns, etc.)
2. **Duplicate codes:** Same code repeated multiple times (42979×4)
3. **Frequency bias:** Model defaults to common codes (42979) regardless of age

**Root Cause:**
- One-way conditioning P(codes | age, sex) without reverse constraint
- No gradient signal preventing medically impossible codes
- Manual sampling loop missing duplicate suppression

**Solution:**
- Multi-task learning with auxiliary age/sex prediction
- Switch to `model.generate()` with `no_repeat_ngram_size=1`
- Remove race from demographics (medical + statistical + ethical reasons)

**Expected Improvement:**
- Age violations: 30% → <2%
- Sex violations: Unknown → <1%
- Duplicates: High → ~0%
- Jaccard: 0.403 → 0.42-0.45
