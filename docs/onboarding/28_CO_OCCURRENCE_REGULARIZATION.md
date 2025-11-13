# 28: Co-occurrence Regularization - Explicit Penalty for Rare Code Pairs

**Estimated Time:** 75 minutes
**Prerequisites:** [14_LOSS_FUNCTIONS.md](14_LOSS_FUNCTIONS.md), [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)
**Next:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)

---

## Learning Objectives

- Understand co-occurrence matrix construction (pairwise counts)
- Learn co-occurrence regularization loss (penalty for rare pairs)
- Understand threshold=5 (rare vs common pairs)
- Learn loss weight=0.05 (balance coherence and validity)
- Understand why this is a novel contribution (not in PromptEHR)

---

## The Semantic Coherence Problem

### Observed Issue

**Baseline model (no co-occurrence loss):**
- Medical validity: 99% (age/sex appropriate)
- JS divergence: 0.61 (high, target < 0.3)
- Co-occurrence score: 3.39 (low, target > 20)
- Top-100 overlap: 0.04 (very low, target > 0.5)

**Problem:** Generated codes are medically valid but statistically implausible.

**Example:**
```
Generated patient: 65M, ["401.9", "799.3", "263.9", "557.1"]
# Codes exist in training data, but never co-occur together
# Real-world: "401.9" (hypertension) often co-occurs with "250.00" (diabetes)
# But model generates rare combinations
```

### Root Cause

**Language model loss optimizes:**
- Token-level likelihood: P(code_i | context)
- NOT co-occurrence likelihood: P(code_i, code_j appear together)

**Result:** Model generates valid codes but ignores pairwise statistics.

---

## Co-occurrence Matrix

### Construction (cooccurrence_utils.py:21-100)

```python
def build_cooccurrence_matrix(
    training_patients: list,
    vocabulary,
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """Build pairwise code co-occurrence matrix from training data.

    Returns:
        Symmetric matrix [vocab_size, vocab_size] with co-occurrence counts
    """
```

**Algorithm:**
1. For each patient in training data
2. For each visit in patient
3. For each pair of codes (i, j) in visit
4. Increment matrix[i, j] += 1

**Example:**

**Training data:**
```python
Patient 1:
  Visit 1: ["401.9", "250.00", "585.9"]  # 3 codes → 3 pairs
  Visit 2: ["428.0", "401.9"]            # 2 codes → 1 pair

Patient 2:
  Visit 1: ["401.9", "250.00"]           # 2 codes → 1 pair
```

**Co-occurrence counts:**
```python
# From Visit 1 (Patient 1):
matrix["401.9", "250.00"] += 1
matrix["401.9", "585.9"] += 1
matrix["250.00", "585.9"] += 1

# From Visit 2 (Patient 1):
matrix["428.0", "401.9"] += 1

# From Visit 1 (Patient 2):
matrix["401.9", "250.00"] += 1  # Increment again

# Final matrix:
matrix["401.9", "250.00"] = 2  # Co-occurred in 2 visits
matrix["401.9", "585.9"] = 1
matrix["250.00", "585.9"] = 1
matrix["428.0", "401.9"] = 1
```

### Matrix Properties

**Symmetry:**
```python
matrix[i, j] == matrix[j, i]
# Order doesn't matter: "401.9" with "250.00" = "250.00" with "401.9"
```

**Sparsity (MIMIC-III with 6,985 codes):**
```
Possible pairs: 6,985 × 6,984 / 2 = 24.4M
Observed pairs: ~696K (2.9% coverage)
Sparse matrix: 97.1% zeros
```

**Distribution:**
- Top 1% of pairs: Account for 60% of all co-occurrences
- Most pairs (70%): Co-occur < 5 times (rare)
- Long tail: Many codes never co-occur

---

## Co-occurrence Regularization Loss

### Motivation

**Goal:** Penalize model for generating rare code pairs

**Example:**
```python
Generated visit: ["401.9", "799.3"]
# Check co-occurrence:
matrix["401.9", "799.3"] = 2  # Only co-occurred 2 times (rare!)

# Penalty: This pair should have low probability
```

### Loss Formula (cooccurrence_utils.py:180-280)

```python
def cooccurrence_loss(
    code_sequences: torch.Tensor,     # [batch, seq_len]
    cooccur_matrix: torch.Tensor,     # [vocab_size, vocab_size]
    threshold: int = 5                # Rare pair threshold
) -> torch.Tensor:
    """Compute co-occurrence regularization loss.

    Args:
        code_sequences: Generated code token IDs
        cooccur_matrix: Pairwise co-occurrence counts from training
        threshold: Pairs with count < threshold are penalized

    Returns:
        Scalar loss (average penalty across batch)
    """
```

**Step 1: Extract code pairs from generated sequence**

```python
# Example sequence (token IDs):
# [1, 3, 7, 8, 9, 4, 2]
# <BOS> <v> 401.9 250.00 428.0 <\v> <EOS>

# Extract only diagnosis codes (ignore special tokens):
codes = [7, 8, 9]  # 401.9, 250.00, 428.0

# Generate all pairs:
pairs = [(7, 8), (7, 9), (8, 9)]
#        (401.9, 250.00), (401.9, 428.0), (250.00, 428.0)
```

**Step 2: Look up co-occurrence counts**

```python
# For each pair (i, j):
count_7_8 = cooccur_matrix[7, 8]  # How many times 401.9 and 250.00 co-occurred
count_7_9 = cooccur_matrix[7, 9]
count_8_9 = cooccur_matrix[8, 9]

# Example values:
# count_7_8 = 45  (common pair)
# count_7_9 = 12  (moderate)
# count_8_9 = 2   (rare!)
```

**Step 3: Compute penalty for rare pairs**

```python
# Rare pair: count < threshold (default: 5)
is_rare = (cooccur_matrix < threshold).float()

# Penalty for rare pairs only
penalty = 0.0
for (i, j) in pairs:
    if cooccur_matrix[i, j] < threshold:
        penalty += 1.0  # Fixed penalty per rare pair

# Example:
# pair (7, 8): count=45 ≥ 5 → no penalty
# pair (7, 9): count=12 ≥ 5 → no penalty
# pair (8, 9): count=2 < 5 → penalty += 1.0

# Total penalty = 1.0
```

**Step 4: Normalize by number of pairs**

```python
num_pairs = len(pairs)
loss = penalty / num_pairs

# Example: 1.0 / 3 = 0.333
```

### Loss Weight in Total Loss

```python
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccur_loss
```

**Why weight=0.05?**
- Higher than auxiliary losses (0.001) → Co-occurrence more important
- Lower than LM loss (1.0) → Don't destroy generation quality
- Balance: Improve coherence while maintaining validity

**Ablation studies:**
- weight=0.0: JS divergence=0.61, co-occurrence=3.39 (poor coherence)
- weight=0.01: JS divergence=0.45, co-occurrence=12 (moderate improvement)
- weight=0.05: JS divergence=0.35, co-occurrence=18 (target: good balance)
- weight=0.1: JS divergence=0.25, co-occurrence=25 (better coherence, but lower medical validity)

---

## Implementation Details

### Sparse Matrix Optimization

**Challenge:** 6,985 × 6,985 = 48.8M elements (195 MB as float32)

**Solution:** Sparse CSR matrix (scipy) stores only non-zero values
```python
import scipy.sparse as sp

# Convert to sparse CSR format
cooccur_sparse = sp.csr_matrix(cooccur_matrix)
# Storage: Only ~696K observed pairs (vs 48.8M total)
# Memory: ~3 MB vs 195 MB (65× reduction)
```

**During training:**
```python
# Convert batch codes to pairs
# Lookup counts from sparse matrix (efficient)
# Compute penalty (vectorized operations)
```

### Threshold Selection

**Threshold=5 rationale:**
- Pairs with count < 5: Likely spurious co-occurrences
- Pairs with count ≥ 5: Statistically significant patterns
- Trade-off: Too low (1-2) → Many rare pairs, slow training
             Too high (10-20) → Ignore moderate patterns

**Distribution analysis:**
```
Threshold | % Rare Pairs | % Common Pairs
----------|--------------|---------------
1         | 0%           | 100%
2         | 30%          | 70%
5         | 70%          | 30%   ← Selected
10        | 85%          | 15%
20        | 92%          | 8%
```

**Threshold=5:** Captures top 30% most common patterns, penalizes bottom 70%.

---

## Integration with Hierarchical Training

### Category-Level Co-occurrence

**Challenge:** Flat codes (6,985) → Sparse co-occurrence (2.9% coverage)

**Solution:** Category-level co-occurrence (943 categories)
```
Possible pairs: 943 × 942 / 2 = 444K
Observed pairs: ~340K (76% coverage)  # 26× improvement!
```

**Benefits:**
- Denser matrix → More reliable statistics
- Better generalization → Unseen code pairs benefit from category patterns
- Faster training → Fewer rare pairs to penalize

**See:** [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md) for hierarchical approach

---

## Example: Full Loss Computation

### Generated Batch

```python
batch = {
    'input_ids': [...],
    'labels': [[1, 3, 7, 8, 9, 4, 2]],  # <BOS> <v> 401.9 250.00 428.0 <\v> <EOS>
    'x_num': [[65.0]],
    'x_cat': [[0]]
}
```

### Forward Pass

**Step 1: LM loss**
```python
logits = model(input_ids, x_num, x_cat)
lm_loss = CrossEntropyLoss(logits, labels)
# lm_loss = 2.45
```

**Step 2: Auxiliary losses**
```python
age_pred = age_head(decoder_output)
sex_pred = sex_head(decoder_output)
age_loss = MSELoss(age_pred, 65.0)
sex_loss = CrossEntropyLoss(sex_pred, 0)
# age_loss = 12.3
# sex_loss = 0.89
```

**Step 3: Co-occurrence loss**
```python
# Extract codes from labels: [7, 8, 9] (401.9, 250.00, 428.0)
# Generate pairs: (7,8), (7,9), (8,9)

# Look up counts:
cooccur_matrix[7, 8] = 45  # Common (≥5)
cooccur_matrix[7, 9] = 12  # Common (≥5)
cooccur_matrix[8, 9] = 2   # Rare (<5) → penalty!

# Penalty: 1 rare pair out of 3 total
cooccur_loss = 1 / 3 = 0.333
```

**Step 4: Total loss**
```python
total_loss = 2.45 + 0.001*12.3 + 0.001*0.89 + 0.05*0.333
           = 2.45 + 0.012 + 0.001 + 0.017
           = 2.480
```

**Loss breakdown:**
- LM loss: 98.8% (dominates)
- Age loss: 0.5%
- Sex loss: 0.04%
- Co-occurrence loss: 0.7% (higher than auxiliary, but still small)

---

## Results and Impact

### Before Co-occurrence Loss

**Metrics (baseline model):**
- JS divergence: 0.61 (high, poor distribution match)
- Co-occurrence score: 3.39 (low, unrealistic pairs)
- Top-100 overlap: 0.04 (very low, wrong frequent codes)
- Medical validity: 99% (age/sex appropriate)

### After Co-occurrence Loss (weight=0.05)

**Metrics (with co-occurrence regularization):**
- JS divergence: 0.35 → 43% improvement
- Co-occurrence score: 18.2 → 5.4× improvement
- Top-100 overlap: 0.28 → 7× improvement
- Medical validity: 97% (slight decrease, acceptable trade-off)

**Conclusion:** Co-occurrence loss dramatically improves semantic coherence with minimal impact on medical validity.

---

## Try It Yourself

### Exercise 1: Build Co-occurrence Matrix

```python
from cooccurrence_utils import build_cooccurrence_matrix
from data_loader import load_mimic_data
import logging

logger = logging.getLogger(__name__)
patients, vocab = load_mimic_data(..., num_patients=1000)

# Build matrix
cooccur_matrix = build_cooccurrence_matrix(patients, vocab, logger)

# Inspect statistics
print(f"Matrix shape: {cooccur_matrix.shape}")
print(f"Non-zero entries: {(cooccur_matrix > 0).sum()}")
print(f"Sparsity: {(cooccur_matrix == 0).sum() / cooccur_matrix.numel() * 100:.1f}%")

# Most common pair
max_count = cooccur_matrix.max()
i, j = (cooccur_matrix == max_count).nonzero()[0]
code_i = vocab.idx2code[i.item()]
code_j = vocab.idx2code[j.item()]
print(f"Most common pair: {code_i} + {code_j} ({max_count} times)")
```

[IN PROGRESS - Additional exercises on loss computation and threshold analysis]

---

## Summary

**Co-occurrence regularization explicitly penalizes rare code pairs:**

1. **Co-occurrence Matrix:** [vocab_size, vocab_size] pairwise counts from training data
2. **Sparsity:** Only 2.9% of possible pairs observed in MIMIC-III
3. **Rare Pair Threshold:** Pairs with count < 5 are penalized
4. **Loss:** Average penalty per rare pair in generated sequence
5. **Weight:** 0.05 (balance coherence and validity)
6. **Impact:** 43% improvement in JS divergence, 5.4× better co-occurrence score
7. **Novel Contribution:** Not present in PromptEHR baseline

**Key Files:**
- `cooccurrence_utils.py:21-100` - Matrix construction
- `cooccurrence_utils.py:180-280` - Loss computation
- `trainer_hierarchical.py:200-250` - Integration in training loop

---

## What's Next?

**Next:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md) - Dual vocabulary (categories + codes), hierarchical token sequences

**Alternative:**
- [27_HIERARCHICAL_GENERATION.md](27_HIERARCHICAL_GENERATION.md) - Two-stage generation (category → code)
- [31_HIERARCHICAL_DATASET.md](31_HIERARCHICAL_DATASET.md) - Hierarchical data pipeline

---

**Navigation:**
- ← Back to [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)
- → Next: [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
