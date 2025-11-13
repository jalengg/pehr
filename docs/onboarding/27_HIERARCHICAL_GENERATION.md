# 27: Hierarchical Generation - Two-Stage Category→Code Approach

**Estimated Time:** 60 minutes
**Prerequisites:** [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)
**Next:** [28_CO_OCCURRENCE_REGULARIZATION.md](28_CO_OCCURRENCE_REGULARIZATION.md)

---

## Learning Objectives

- Understand two-stage generation: Stage 1 (categories) → Stage 2 (codes)
- Learn category-level training (943 categories vs 6,985 codes)
- Understand sparsity reduction (7.4×) and co-occurrence improvement (18.6×)
- Learn code sampling within categories

---

## Two-Stage Generation

### Stage 1: Generate Category Sequence

```python
# Input: Demographics (age=65, gender=M)
# Output: Category sequence
categories = model.generate(x_num=[65.0], x_cat=[0])
# Result: ["401", "250", "585"]  (category codes)
```

### Stage 2: Sample Specific Codes

```python
# For each category, sample specific code
codes = []
for category in categories:
    # Get all codes in this category
    codes_in_category = hierarchy.category_to_codes[category]
    # Sample one code (uniform or frequency-weighted)
    code = random.choice(codes_in_category)
    codes.append(code)

# Result: ["401.9", "250.00", "585.9"]  (full ICD-9 codes)
```

### Why Two-Stage?

**Sparsity reduction:**
- Flat: 6,985 codes → 48.7M possible pairs, 1.4% coverage
- Hierarchical: 943 categories → 444K possible pairs, 76% coverage
- **Improvement: 54× better coverage**

**Co-occurrence improvement:**
- Flat: Average 1.4 co-occurring pairs per code
- Hierarchical: Average 26 co-occurring pairs per category
- **Improvement: 18.6× more co-occurrence examples**

---

## Implementation

[IN PROGRESS - See hierarchical_generation.py for two-stage generation]

**Key points:**
- Stage 1: Model trained on category sequences
- Stage 2: Deterministic or stochastic code sampling
- Handles unseen code combinations via category-level patterns

**See:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md) for tokenization details

---

**Navigation:** [← 29](29_ICD9_HIERARCHY.md) | [→ 28](28_CO_OCCURRENCE_REGULARIZATION.md) | [↑ Index](00_START_HERE.md)
