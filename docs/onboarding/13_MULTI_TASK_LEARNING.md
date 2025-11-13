# 13: Multi-Task Learning - Age/Sex Prediction for Medical Validity

**Estimated Time:** 60 minutes
**Prerequisites:** [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md)
**Next:** [14_LOSS_FUNCTIONS.md](14_LOSS_FUNCTIONS.md)

---

## Learning Objectives

- Understand why auxiliary tasks improve medical validity
- Learn age prediction head (regression, MSE loss)
- Learn sex prediction head (classification, cross-entropy loss)
- Understand loss weight balancing (0.001 for both)
- Learn why low weights preserve semantic coherence

---

## Why Multi-Task Learning?

**Primary task:** Generate medically valid ICD-9 codes
**Challenge:** Age/sex-inappropriate codes (pregnancy codes for males)

**Solution:** Add auxiliary prediction heads
- Age head: Predict patient age from generated codes
- Sex head: Predict patient sex from generated codes
- Forces model to encode demographics in generated sequences

**Hypothesis:** If model can predict age/sex from codes, it learned to generate age/sex-appropriate codes.

---

## Age Prediction Head

### Architecture

```python
class AgePredict Head(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)  # Regression

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, 768] decoder output

        Returns:
            [batch, 1] age predictions
        """
        # Use <EOS> token representation
        eos_hidden = hidden_states[:, -1, :]  # [batch, 768]
        age_pred = self.linear(eos_hidden)    # [batch, 1]
        return age_pred
```

**Loss:**
```python
age_loss = nn.MSELoss()(age_pred, true_age)
```

---

## Sex Prediction Head

### Architecture

```python
class SexPredictionHead(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)  # Classification (M/F)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, 768]

        Returns:
            [batch, 2] class logits
        """
        eos_hidden = hidden_states[:, -1, :]
        sex_logits = self.linear(eos_hidden)  # [batch, 2]
        return sex_logits
```

**Loss:**
```python
sex_loss = nn.CrossEntropyLoss()(sex_logits, true_sex)
```

---

## Loss Weight Balancing

### Total Loss Formula

```python
total_loss = lm_loss + alpha * age_loss + beta * sex_loss
```

**Current weights (config.py):**
- `alpha = 0.001` (age weight)
- `beta = 0.001` (sex weight)

**Why 0.001?**
- LM loss dominates (learns realistic code distributions)
- Auxiliary losses provide weak guidance (medical validity)
- High weights (0.01-0.2) destroy semantic coherence

**Historical evolution:**
- Initial: alpha=0.01, beta=0.2 → Medical validity 99%, but JS divergence=0.61 (poor coherence)
- Current: alpha=0.001, beta=0.001 → Balances validity and coherence

---

## Implementation

[IN PROGRESS - See prompt_bart_model.py:160-220 for multi-task head integration]

**Key points:**
- Heads applied to decoder final layer output
- Use <EOS> token representation (aggregates full sequence info)
- Gradients from auxiliary losses flow through decoder

---

## Summary

**Multi-task learning improves medical validity:**

1. **Age Head:** Regression (768 → 1), MSE loss
2. **Sex Head:** Classification (768 → 2), cross-entropy loss
3. **Loss Weights:** 0.001 for both (weak supervision)
4. **EOS Token:** Used as sequence representation for predictions
5. **Trade-off:** Medical validity vs semantic coherence

**Key Files:**
- `prompt_bart_model.py:160-220` - Multi-task head implementation
- `config.py` - Loss weight configuration

---

**Navigation:**
- ← Back to [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md)
- → Next: [14_LOSS_FUNCTIONS.md](14_LOSS_FUNCTIONS.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
