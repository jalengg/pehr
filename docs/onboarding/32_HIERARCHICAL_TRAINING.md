# 32: Hierarchical Training Loop - Category-Level Training with Co-occurrence Loss

**Estimated Time:** 60 minutes
**Prerequisites:** [31_HIERARCHICAL_DATASET.md](31_HIERARCHICAL_DATASET.md), [28_CO_OCCURRENCE_REGULARIZATION.md](28_CO_OCCURRENCE_REGULARIZATION.md)
**Next:** [33_HIERARCHICAL_EVALUATION.md](33_HIERARCHICAL_EVALUATION.md)

---

## Learning Objectives

- Understand trainer_hierarchical.py structure
- Learn category-level co-occurrence matrix construction
- Understand loss integration: LM + age + sex + co-occurrence
- Learn how hierarchical training improves semantic coherence

---

## Hierarchical Training Pipeline

### Key Differences from Flat Training

| Aspect | Flat (trainer.py) | Hierarchical (trainer_hierarchical.py) |
|--------|-------------------|----------------------------------------|
| Vocabulary | 6,985 codes | 943 categories |
| Tokenizer | DiagnosisCodeTokenizer | HierarchicalDiagnosisTokenizer |
| Dataset | EHRPatientDataset | HierarchicalEHRDataset |
| Co-occurrence | No | Yes (category-level) |
| Loss | LM + age + sex | LM + age + sex + cooccur |

### Training Loop (trainer_hierarchical.py)

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            x_num=batch['x_num'],
            x_cat=batch['x_cat']
        )

        # LM loss (from model)
        lm_loss = outputs.loss

        # Auxiliary losses
        age_loss = age_criterion(outputs.age_pred, batch['x_num'])
        sex_loss = sex_criterion(outputs.sex_pred, batch['x_cat'])

        # Co-occurrence loss (novel)
        cooccur_loss = cooccurrence_loss_efficient(
            batch['labels'],
            cooccur_matrix,
            tokenizer,
            threshold=5
        )

        # Total loss
        total_loss = (lm_loss +
                      0.001 * age_loss +
                      0.001 * sex_loss +
                      0.05 * cooccur_loss)

        # Backward pass
        total_loss.backward()
        optimizer.step()
```

### Category-Level Co-occurrence Matrix

**Construction:**
```python
# Build category-level co-occurrence from training data
category_cooccur = build_cooccurrence_matrix(
    train_patients,
    hierarchy.category_vocabulary  # 943 categories
)

# Result:
# Shape: [943, 943]
# Coverage: 76% (vs 1.4% for code-level)
# Observed pairs: 340K (vs 696K code-level, but 444K possible vs 48.7M)
```

**Benefits:**
- Denser matrix (more reliable statistics)
- Faster training (fewer rare pairs to penalize)
- Better generalization (category patterns transfer to unseen codes)

---

## Results

**Hierarchical model (with co-occurrence loss):**
- JS divergence: 0.29 (target: <0.3) ✓
- Co-occurrence score: 21.5 (target: >20) ✓
- Top-100 overlap: 0.52 (target: >0.5) ✓
- Medical validity: 97% (acceptable trade-off)

**Comparison to baseline:**
- Semantic coherence: 52% improvement in JS divergence
- Co-occurrence: 6.3× improvement
- Medical validity: -2% (minor decrease)

---

[IN PROGRESS - Full training loop details, checkpointing, and evaluation integration]

**See:** trainer_hierarchical.py for complete implementation

---

**Navigation:** [← 31](31_HIERARCHICAL_DATASET.md) | [→ 33](33_HIERARCHICAL_EVALUATION.md) | [↑ Index](00_START_HERE.md)
