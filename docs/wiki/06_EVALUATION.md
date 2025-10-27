# Evaluation

**Last Updated:** October 24, 2025

This document describes how to evaluate generated synthetic patients for medical validity and semantic coherence.

## Evaluation Framework

**Two complementary approaches:**

1. **Medical Validity** - Are generated codes clinically appropriate?
2. **Semantic Coherence** - Do generated codes match training data distributions?

## 1. Medical Validity (evaluate_medical_validity.py)

### Purpose

Assess whether generated codes are medically appropriate for patient demographics.

### Metrics

#### 1.1 Duplicate Code Rate

**Definition:** Percentage of duplicate codes within a single visit

**Target:** <1%

**Why Important:** Clinically implausible to diagnose same code twice in one admission

**Implementation:**
```python
def check_duplicate_codes(visit: List[str]) -> Dict[str, int]:
    code_counts = Counter(visit)
    duplicates = {code: count for code, count in code_counts.items() if count > 1}
    return duplicates

duplicate_rate = num_duplicates / total_codes * 100
```

**Pass Criteria:** ✓ if duplicate_rate < 1%

#### 1.2 Age-Inappropriate Codes

**Definition:** Percentage of codes inappropriate for patient age

**Target:** <2% (>98% appropriate)

**Rules:** (See docs/reference/medical_validity.md for full list)

```python
# Neonatal-only codes (age 0-1 only)
NEONATAL_ONLY = ['V30', 'V31', '76', '77', 'V502', 'V290']

# Adult minimum ages
ADULT_MINIMUM_AGES = {
    '42979': 18,  # Atrial fibrillation
    '41401': 30,  # Coronary atherosclerosis
    '412': 30,    # Old MI
    '2720': 18,   # Hypercholesterolemia
    ...
}

def is_age_appropriate(code: str, age: float) -> bool:
    # Neonatal codes
    if age > 1:
        for prefix in NEONATAL_ONLY:
            if code.startswith(prefix):
                return False

    # Adult minimum age requirements
    if code in ADULT_MINIMUM_AGES:
        if age < ADULT_MINIMUM_AGES[code]:
            return False

    return True
```

**Pass Criteria:** ✓ if age_inappropriate_rate < 2%

#### 1.3 Sex-Inappropriate Codes

**Definition:** Percentage of codes inappropriate for patient sex

**Target:** <1% (>99% appropriate)

**Rules:**

```python
# Male-only codes
MALE_ONLY_CODES = ['V502']  # Circumcision

# Female-only codes (pregnancy/childbirth)
FEMALE_ONLY_PREFIXES = ['V27', '640', '641', '642', ..., '659']

def is_sex_appropriate(code: str, sex: str) -> bool:
    # Male-only codes
    if sex == 'F' and code in MALE_ONLY_CODES:
        return False

    # Female-only codes
    if sex == 'M':
        for prefix in FEMALE_ONLY_PREFIXES:
            if code.startswith(prefix):
                return False

    return True
```

**Pass Criteria:** ✓ if sex_inappropriate_rate < 1%

#### 1.4 Jaccard Similarity (Conditional Only)

**Definition:** Overlap between generated and target codes

**Target:** 0.40-0.45 for prompt_prob=0.5

**Formula:**
```python
def compute_jaccard_similarity(generated: List[str], target: List[str]) -> float:
    gen_set = set(generated)
    tgt_set = set(target)

    intersection = len(gen_set & tgt_set)
    union = len(gen_set | tgt_set)

    return intersection / union if union > 0 else 0.0
```

**Interpretation:**
- **High (>0.6):** Overfitting (generates exact training data)
- **Medium (0.40-0.45):** Good (captures semantics, not overfitting)
- **Low (<0.3):** Poor reconstruction

**Pass Criteria:** ✓ if 0.40 ≤ jaccard ≤ 0.50

### Running Medical Validity Evaluation

```bash
python evaluate_medical_validity.py
```

**Output:**
```
Medical Validity Evaluation
================================================================================
Generation Statistics:
  Patients evaluated: 100
  Total visits: 131
  Total codes: 1203
  Avg codes per visit: 9.18

Medical Validity:
  Duplicate codes: 0 / 1203 (0.00%)
  Age-inappropriate: 18 / 1203 (1.50%)
  Sex-inappropriate: 12 / 1203 (1.00%)

Reconstruction Quality:
  Mean Jaccard similarity: 0.4234 ± 0.0782

Pass/Fail Summary:
  ✓ Duplicate suppression
  ✓ Age appropriateness
  ✗ Sex appropriateness (1.00% at threshold, marginal)
  ✓ Jaccard similarity
```

## 2. Semantic Coherence (evaluate_semantic_coherence.py)

### Purpose

Assess whether generated codes match training data distributions (not just medical validity).

**Key Insight:** Jaccard measures exact match, not semantic plausibility. Model can generate medically valid but statistically implausible code combinations.

### Metrics

#### 2.1 Code Frequency Divergence (JS Divergence)

**Definition:** Jensen-Shannon divergence between generated and training code frequency distributions

**Target:** <0.3 (lower is better)

**Formula:**
```python
def compute_code_frequency_divergence(generated_patients, training_patients) -> float:
    # Count code frequencies
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_counts = Counter(gen_codes)
    train_counts = Counter(train_codes)

    # Create probability distributions
    all_codes = set(gen_counts.keys()) | set(train_counts.keys())
    gen_probs = np.array([gen_counts.get(code, 0) + 1 for code in all_codes])
    train_probs = np.array([train_counts.get(code, 0) + 1 for code in all_codes])

    gen_probs = gen_probs / gen_probs.sum()
    train_probs = train_probs / train_probs.sum()

    # Jensen-Shannon divergence
    return float(jensenshannon(gen_probs, train_probs))
```

**Interpretation:**
- **Excellent (<0.1):** Distributions very similar
- **Good (0.1-0.3):** Distributions reasonably similar
- **Poor (>0.3):** Distributions differ significantly

**Pass Criteria:** ✓ if JS < 0.3

#### 2.2 Distribution Match Tests (KS Tests)

**Definition:** Kolmogorov-Smirnov tests comparing distributions

**Target:** p-value > 0.05 (distributions match)

**Tests:**
1. Visits per patient distribution
2. Codes per visit distribution

```python
def compute_distribution_match(generated_patients, training_patients) -> dict:
    # Visits per patient
    gen_visits = [len(p.visits) for p in generated_patients]
    train_visits = [len(p.visits) for p in training_patients]
    visits_stat, visits_pvalue = ks_2samp(gen_visits, train_visits)

    # Codes per visit
    gen_codes = [len(visit) for p in generated_patients for visit in p.visits]
    train_codes = [len(visit) for p in training_patients for visit in p.visits]
    codes_stat, codes_pvalue = ks_2samp(gen_codes, train_codes)

    return {
        'visits_statistic': visits_stat,
        'visits_pvalue': visits_pvalue,
        'codes_statistic': codes_stat,
        'codes_pvalue': codes_pvalue
    }
```

**Pass Criteria:** ✓ if both p-values > 0.05

#### 2.3 Top-K Code Overlap

**Definition:** Jaccard similarity of top-100 most common codes

**Target:** >0.5

**Formula:**
```python
def compute_top_k_overlap(generated_patients, training_patients, k=100) -> float:
    # Get top-K codes
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_top_k = set([code for code, _ in Counter(gen_codes).most_common(k)])
    train_top_k = set([code for code, _ in Counter(train_codes).most_common(k)])

    # Jaccard
    intersection = len(gen_top_k & train_top_k)
    union = len(gen_top_k | train_top_k)

    return intersection / union if union > 0 else 0.0
```

**Interpretation:**
- **Excellent (>0.7):** Model learned most common codes
- **Good (0.5-0.7):** Model learned many common codes
- **Poor (<0.5):** Model missing common codes

**Pass Criteria:** ✓ if overlap > 0.5

#### 2.4 Pairwise Co-occurrence Score

**Definition:** Average co-occurrence frequency of generated code pairs in training data

**Target:** >20

**Formula:**
```python
def compute_cooccurrence_score(generated_patients, training_patients) -> float:
    # Build co-occurrence matrix from training
    cooccur = defaultdict(int)
    for patient in training_patients:
        for visit in patient.visits:
            for i, code1 in enumerate(visit):
                for code2 in visit[i+1:]:
                    pair = tuple(sorted([code1, code2]))
                    cooccur[pair] += 1

    # Score generated pairs
    scores = []
    for patient in generated_patients:
        for visit in patient.visits:
            for i, code1 in enumerate(visit):
                for code2 in visit[i+1:]:
                    pair = tuple(sorted([code1, code2]))
                    scores.append(cooccur.get(pair, 0))

    return np.mean(scores) if scores else 0.0
```

**Interpretation:**
- **Excellent (>50):** Codes frequently co-occur in training
- **Good (20-50):** Codes sometimes co-occur in training
- **Poor (<20):** Codes rarely co-occur together

**Pass Criteria:** ✓ if score > 20

### Running Semantic Coherence Evaluation

```bash
python evaluate_semantic_coherence.py
```

**Output:**
```
Semantic Coherence Evaluation
================================================================================
Generating 100 patients (zero-prompt, sampled structure)
...

Computing Semantic Coherence Metrics
================================================================================

1. Code Frequency Distribution Match...
   Jensen-Shannon Divergence: 0.6064
   ✗ Poor - distributions differ significantly

2. Distribution Match Tests...
   Visits per patient: p=0.9302, stat=0.0527
   ✓ Match - distributions are similar
   Codes per visit: p=0.9388, stat=0.0454
   ✓ Match - distributions are similar

3. Top-100 Code Overlap...
   Jaccard similarity: 0.0382
   ✗ Poor - missing common codes

4. Pairwise Co-occurrence Analysis...
   Average co-occurrence count: 3.39
   ✗ Poor - codes rarely co-occur together

SUMMARY
================================================================================
Code Frequency Divergence: 0.6064 (Poor)
Visit Distribution Match: p=0.9302 (Match)
Codes Distribution Match: p=0.9388 (Match)
Top-100 Overlap: 0.0382 (Poor)
Co-occurrence Score: 3.39 (Poor)

Interpretation: Generated data needs improvement
```

## 3. Comparison: Medical Validity vs Semantic Coherence

| Aspect | Medical Validity | Semantic Coherence |
|--------|-----------------|-------------------|
| **Question** | "Are codes appropriate?" | "Are codes realistic?" |
| **Focus** | Demographics compatibility | Training distribution fidelity |
| **Example Pass** | No pregnancy codes for males | Top-100 codes match training |
| **Example Fail** | Pediatric codes for elderly | Rare code combinations |
| **Trade-off** | Can have high validity but poor coherence | Can have high coherence but poor validity |

**Current Model (aux weights 0.001):**
- Medical validity: Good (99% age, 96% sex)
- Semantic coherence: Poor (JS 0.61, overlap 0.04)

**Previous Model (aux weights 0.01-0.2):**
- Medical validity: Excellent (99% age, 99% sex)
- Semantic coherence: Very poor (JS 0.65, overlap 0.02)

## 4. Interpretation Guide

### Green Light (Production-Ready)

```
Medical Validity:
  ✓ Duplicate rate < 1%
  ✓ Age appropriate > 98%
  ✓ Sex appropriate > 99%
  ✓ Jaccard 0.40-0.45

Semantic Coherence:
  ✓ JS divergence < 0.3
  ✓ Distribution match (both p > 0.05)
  ✓ Top-100 overlap > 0.5
  ✓ Co-occurrence > 20
```

### Yellow Light (Needs Tuning)

```
Partial success in either medical validity OR semantic coherence
→ Adjust auxiliary loss weights
→ Try curriculum learning
→ Apply post-processing filters
```

### Red Light (Major Issues)

```
Failures in both medical validity AND semantic coherence
→ Retrain with different architecture
→ Check data quality
→ Review model configuration
```

## 5. Debugging Poor Performance

### Poor Medical Validity

**Symptoms:** >5% age/sex violations

**Diagnosis:**
1. Check auxiliary loss weights too low
2. Check model loaded correctly
3. Check generation temperature too high

**Solutions:**
- Increase auxiliary loss weights (0.001 → 0.005)
- Apply post-processing filters
- Lower temperature (0.7 → 0.3)

### Poor Semantic Coherence

**Symptoms:** JS > 0.5, top-100 overlap < 0.2

**Diagnosis:**
1. Auxiliary losses dominating LM loss
2. Training data too small
3. Model underfitting

**Solutions:**
- Reduce auxiliary loss weights (0.01 → 0.001)
- Increase training data (25k → 50k patients)
- Train longer (30 → 50 epochs)

## Next Steps

- **Improve generation:** See [Generation](05_GENERATION.md)
- **Understand metrics:** See docs/reference/medical_validity.md
- **Learn usage:** See [Usage Guide](07_USAGE_GUIDE.md)
