# 12: Conditional Prompt - Demographic Conditioning via Reparameterization

**Estimated Time:** 75 minutes
**Prerequisites:** [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md)
**Next:** [13_MULTI_TASK_LEARNING.md](13_MULTI_TASK_LEARNING.md)

---

## Learning Objectives

- Understand ConditionalPrompt architecture (numerical + categorical encoding)
- Master reparameterization trick: `weight * value + bias` in d_hidden=128 space
- Learn why d_hidden=128 intermediate dimension improves gradient flow
- Understand offset-based indexing for categorical features (prevents collision)
- Learn how prompts are prepended to encoder/decoder inputs
- Understand why dual prompt encoders (encoder + decoder) are used

---

## The Demographic Conditioning Problem

### Challenge

**Goal:** Condition EHR generation on demographics (age, gender)
- Age: Continuous variable (0-90 years)
- Gender: Categorical variable (M/F)

**Requirements:**
1. Convert demographics → embedding vectors (768-dim)
2. Inject into encoder and decoder (before code tokens)
3. Ensure gradients flow back to demographic parameters
4. Prevent overfitting (regularization)

**Naive approach (DOESN'T WORK):**
```python
# Direct linear projection
age_embed = nn.Linear(1, 768)(age)  # 1 → 768
# Problem: Too direct, no regularization, poor gradient flow
```

**Our approach (ConditionalPrompt with reparameterization):**
```python
# Two-stage projection with intermediate dimension
age_embed = reparameterize(age, d_hidden=128)  # 1 → 128
age_embed = project(age_embed, 768)            # 128 → 768
# Benefit: Better gradient flow, regularization, stable training
```

---

## Reparameterization Trick

### What Is Reparameterization?

**Standard embedding:**
```python
# Lookup from table
embedding_table[idx] → vector
```

**Reparameterized embedding (for continuous values):**
```python
# Learned transformation
output = weight * input_value + bias
```

**Why "reparameterization"?**
- Instead of learning embedding directly in high-dim space (768)
- Learn in low-dim space (128), then project up
- More parameters → Better expressiveness
- Intermediate space → Better gradient flow

### Mathematical Formulation

**Input:** Age = 65.0

**Step 1: Reparameterization in d_hidden=128 space**
```
weight: [1, 128]  (learned parameter)
bias:   [1, 128]  (learned parameter)
age:    65.0      (input value)

intermediate = weight * age + bias
             = [w1, w2, ..., w128] * 65.0 + [b1, b2, ..., b128]
             = [w1*65+b1, w2*65+b2, ..., w128*65+b128]
             (128-dimensional vector)
```

**Step 2: Project to output dimension (768)**
```
proj: nn.Linear(128, 768, bias=False)

output = proj(intermediate)
       = [768-dimensional vector]
```

**Total transformation:**
```
age (1D) → intermediate (128D) → output (768D)
```

---

## NumericalConditionalPrompt

### Architecture (conditional_prompt.py:10-65)

```python
class NumericalConditionalPrompt(nn.Module):
    """Embeds continuous numerical features (e.g., age) with reparameterization trick."""

    def __init__(self, n_num_features: int, hidden_dim: int, d_hidden: int = 128, prompt_length: int = 1):
        super().__init__()

        # Reparameterization: learned weight and bias in d_hidden space
        self.weight = nn.Parameter(torch.Tensor(n_num_features, d_hidden))
        self.bias = nn.Parameter(torch.Tensor(n_num_features, d_hidden))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.bias)

        # Project from d_hidden to output dimension
        self.proj = nn.Linear(d_hidden, hidden_dim, bias=False)
```

**Parameters:**
- `n_num_features`: Number of continuous features (1 for age)
- `hidden_dim`: Output dimension (768 for BART)
- `d_hidden`: Intermediate dimension (128)
- `prompt_length`: Number of prompt vectors per feature (1)

**Learnable parameters:**
- `weight`: [1, 128] → 128 parameters
- `bias`: [1, 128] → 128 parameters
- `proj.weight`: [128, 768] → 98,304 parameters
- **Total: 98,560 parameters**

### Forward Pass (conditional_prompt.py:41-65)

```python
def forward(self, x_num: torch.Tensor) -> torch.Tensor:
    """Embed numerical features with reparameterization.

    Args:
        x_num: [batch, n_num_features] continuous values.

    Returns:
        [batch, prompt_length * n_num_features, hidden_dim] embeddings.
    """
    batch_size = x_num.shape[0]

    # Reparameterization: weight * value + bias
    # x_num: [batch, n_num_features]
    # weight: [n_num_features, d_hidden]
    # Result: [batch, n_num_features, d_hidden]
    x = self.weight[None] * x_num[..., None]  # Broadcasting
    x = x + self.bias[None]

    # Project to output dimension
    # x: [batch, n_num_features, d_hidden] → [batch, n_num_features, hidden_dim]
    x = self.proj(x)

    return x
```

### Example: Encoding Age

**Input:**
```python
x_num = torch.tensor([[65.0], [45.0], [82.0]])  # [batch=3, n_num_features=1]
```

**Step 1: Reparameterization**
```python
# weight: [1, 128] (example values)
# bias: [1, 128]

# For patient 1 (age=65.0):
intermediate[0] = weight[0] * 65.0 + bias[0]
# Result: [128-dim vector]

# For patient 2 (age=45.0):
intermediate[1] = weight[0] * 45.0 + bias[0]
# Result: [128-dim vector, different from patient 1]
```

**Step 2: Projection**
```python
# proj: nn.Linear(128, 768)
output = proj(intermediate)
# Result: [3, 1, 768]
#         batch=3, n_num_features=1, hidden_dim=768
```

**Final output shape:** `[3, 1, 768]`

---

## CategoricalConditionalPrompt

### The Category Collision Problem

**Naive approach (DOESN'T WORK):**
```python
# Separate embedding tables for each feature
gender_embedding = nn.Embedding(2, 768)      # M/F
ethnicity_embedding = nn.Embedding(6, 768)   # 6 categories

# Problem: Separate embedding tables, harder to learn relationships
```

**Our approach: Single embedding table with offset-based indexing**

```python
# Single embedding table
total_categories = 2 + 6 = 8
embedding = nn.Embedding(8, 128)  # d_hidden=128

# Offset indices to prevent collision
# Gender:    0 (M) → index 0, 1 (F) → index 1
# Ethnicity: 0 → index 2, 1 → index 3, ..., 5 → index 7
```

**Why offsets matter:**
```python
# Without offsets (WRONG):
gender[0] and ethnicity[0] both map to index 0 → collision!

# With offsets (RIGHT):
gender[0] → index 0
ethnicity[0] → index 0 + 2 (offset) = index 2  (no collision)
```

### Architecture (conditional_prompt.py:68-138)

```python
class CategoricalConditionalPrompt(nn.Module):
    """Embeds categorical features with offset-based indexing and reparameterization."""

    def __init__(self, cat_cardinalities: list[int], hidden_dim: int, d_hidden: int = 128, prompt_length: int = 1):
        super().__init__()

        # Compute offset indices
        # Example: [2, 6] → offsets = [0, 2]
        category_offsets = torch.tensor([0] + cat_cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)

        # Single embedding table for all categories
        total_categories = sum(cat_cardinalities)  # 2 + 6 = 8
        self.embeddings = nn.Embedding(total_categories, d_hidden)

        # Learned bias per feature (not per category)
        self.bias = nn.Parameter(torch.Tensor(len(cat_cardinalities), d_hidden))

        # Project from d_hidden to output dimension
        self.proj = nn.Linear(d_hidden, hidden_dim, bias=False)
```

**Parameters:**
- `cat_cardinalities`: [2] for gender (M/F) - ethnicity removed
- `total_categories`: 2
- `category_offsets`: [0] (only one categorical feature)

**Learnable parameters:**
- `embeddings.weight`: [2, 128] → 256 parameters
- `bias`: [1, 128] → 128 parameters
- `proj.weight`: [128, 768] → 98,304 parameters
- **Total: 98,688 parameters**

### Forward Pass (conditional_prompt.py:110-138)

```python
def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
    """Embed categorical features with offset-based indexing.

    Args:
        x_cat: [batch, n_cat_features] categorical IDs.

    Returns:
        [batch, n_cat_features * prompt_length, hidden_dim] embeddings.
    """
    batch_size = x_cat.shape[0]

    # Add offsets to prevent category collision
    # x_cat: [batch, n_cat_features]
    # category_offsets: [n_cat_features]
    x = self.embeddings(x_cat + self.category_offsets[None])

    # Add learned bias per feature
    x = x + self.bias[None]

    # Project to output dimension
    x = self.proj(x)

    return x
```

### Example: Encoding Gender

**Input:**
```python
x_cat = torch.tensor([[0], [1], [0]])  # [batch=3, n_cat_features=1]
# 0 = Male, 1 = Female
```

**Step 1: Add offsets**
```python
# category_offsets = [0] (only one categorical feature)
x_cat_offset = x_cat + 0 = [[0], [1], [0]]
# No change (offset=0 for first feature)
```

**Step 2: Embedding lookup**
```python
# embeddings: nn.Embedding(2, 128)
# Index 0 (Male) → [128-dim vector A]
# Index 1 (Female) → [128-dim vector B]

embedded = [
    vector_A,  # Patient 1 (Male)
    vector_B,  # Patient 2 (Female)
    vector_A   # Patient 3 (Male)
]
# Shape: [3, 1, 128]
```

**Step 3: Add bias**
```python
# bias: [1, 128] (learned parameter)
embedded = embedded + bias
# Shape: [3, 1, 128]
```

**Step 4: Projection**
```python
# proj: nn.Linear(128, 768)
output = proj(embedded)
# Shape: [3, 1, 768]
```

---

## ConditionalPrompt (Combined)

### Architecture (conditional_prompt.py:141-218)

```python
class ConditionalPrompt(nn.Module):
    """Combined prompt encoder for both numerical and categorical features."""

    def __init__(
        self,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list[int]] = None,
        hidden_dim: int = 768,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        super().__init__()

        if n_num_features is not None and n_num_features > 0:
            self.num_prompt = NumericalConditionalPrompt(n_num_features, hidden_dim, d_hidden, prompt_length)
        else:
            self.num_prompt = None

        if cat_cardinalities is not None and len(cat_cardinalities) > 0:
            self.cat_prompt = CategoricalConditionalPrompt(cat_cardinalities, hidden_dim, d_hidden, prompt_length)
        else:
            self.cat_prompt = None
```

### Forward Pass (conditional_prompt.py:178-206)

```python
def forward(
    self,
    x_num: Optional[torch.Tensor] = None,
    x_cat: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Encode demographics to prompt embeddings."""
    prompts = []

    if x_num is not None and self.num_prompt is not None:
        num_embeds = self.num_prompt(x_num)  # [batch, 1, 768]
        prompts.append(num_embeds)

    if x_cat is not None and self.cat_prompt is not None:
        cat_embeds = self.cat_prompt(x_cat)  # [batch, 1, 768]
        prompts.append(cat_embeds)

    combined_prompts = torch.cat(prompts, dim=1)  # [batch, 2, 768]
    return combined_prompts
```

### Example: Full Demographic Encoding

**Input:**
```python
x_num = torch.tensor([[65.0], [45.0]])  # Age
x_cat = torch.tensor([[0], [1]])        # Gender (M, F)
```

**Processing:**
```python
# Step 1: Encode age
num_embeds = num_prompt(x_num)
# Shape: [2, 1, 768]
# num_embeds[0] = age embedding for 65.0
# num_embeds[1] = age embedding for 45.0

# Step 2: Encode gender
cat_embeds = cat_prompt(x_cat)
# Shape: [2, 1, 768]
# cat_embeds[0] = gender embedding for Male
# cat_embeds[1] = gender embedding for Female

# Step 3: Concatenate
combined = torch.cat([num_embeds, cat_embeds], dim=1)
# Shape: [2, 2, 768]
#        batch=2, num_prompts=2 (age + gender), hidden_dim=768
```

**Result:** Two prompt vectors per patient
- Prompt 1: Age encoding
- Prompt 2: Gender encoding

---

## Prompt Prepending in Encoder/Decoder

### Encoder Prompt Injection

**Without prompts:**
```
Token sequence: <BOS> <v> 401.9 250.00 <\v> <EOS>
Token IDs:      [1, 3, 7, 8, 4, 2]
Token embeds:   [batch, 6, 768]
```

**With demographic prompts:**
```
Prompt embeds:  [batch, 2, 768]  (age + gender)
Token embeds:   [batch, 6, 768]

Combined input: [batch, 8, 768]
                |     |     |
              batch  2+6   768
                     prompts+tokens
```

**Sequence structure:**
```
Position:  0      1      2   3   4    5    6   7
Content:  [age] [gender] <BOS> <v> 401.9 250.00 <\v> <EOS>
```

**Attention:** All tokens (including code tokens) can attend to demographic prompts.

### Decoder Prompt Injection

**Same process as encoder:**
```
Prompt embeds:  [batch, 2, 768]
Decoder tokens: [batch, 6, 768]
Combined:       [batch, 8, 768]
```

**Benefit of dual prompts:**
- Encoder prompts: Help attend to relevant codes given demographics
- Decoder prompts: Help generate age/sex-appropriate codes
- Different parameters → Different learned representations

---

## Why d_hidden=128?

### Gradient Flow

**Direct projection (NO intermediate dimension):**
```
age (1D) ──→ output (768D)
        linear(1, 768)

Gradient: d_loss/d_age = d_loss/d_output * d_output/d_age
# Single gradient path, can vanish
```

**Reparameterization (WITH intermediate dimension):**
```
age (1D) ──→ intermediate (128D) ──→ output (768D)
        reparam                   linear(128, 768)

Gradient: d_loss/d_age = d_loss/d_intermediate * d_intermediate/d_age
# More stable, better gradient flow
```

### Expressiveness

**Parameter count comparison:**

| Approach | Parameters | Expressiveness |
|----------|------------|----------------|
| Direct: linear(1, 768) | 768 | Low (1 parameter per output dim) |
| Reparam: (1→128) + linear(128, 768) | 98,560 | High (128 intermediate dims) |

**More parameters → More expressiveness:**
- Can learn complex age → embedding mappings
- 128 intermediate dimensions capture nonlinear relationships

### Regularization

**d_hidden=128 acts as bottleneck:**
- Forces model to compress demographic info into 128 dims
- Prevents overfitting to specific age/gender combinations
- Improves generalization

**Ablation study (not shown):**
- d_hidden=32: Underfits (too much compression)
- d_hidden=128: Optimal (balance)
- d_hidden=512: Overfits (too much capacity)

---

## Try It Yourself

### Exercise 1: Test Reparameterization

```python
from conditional_prompt import NumericalConditionalPrompt
import torch

# Create numerical prompt encoder
num_prompt = NumericalConditionalPrompt(
    n_num_features=1,    # Age only
    hidden_dim=768,      # BART dimension
    d_hidden=128,        # Intermediate dimension
    prompt_length=1
)

# Test with ages
ages = torch.tensor([[65.0], [45.0], [82.0]])

# Encode
prompts = num_prompt(ages)
print(f"Input shape: {ages.shape}")          # [3, 1]
print(f"Output shape: {prompts.shape}")      # [3, 1, 768]
print(f"Prompt for age 65: {prompts[0, 0, :5]}")  # First 5 dims
```

[IN PROGRESS - Additional exercises on categorical encoding, combined prompts, and gradient analysis]

---

## Summary

**ConditionalPrompt converts demographics to embedding vectors via reparameterization:**

1. **Reparameterization Trick:** `weight * value + bias` in d_hidden=128 space
2. **Two-Stage Projection:** 1D → 128D → 768D (better gradient flow)
3. **Numerical Encoding:** Age → reparameterized embedding
4. **Categorical Encoding:** Gender → offset-based embedding lookup
5. **Offset-Based Indexing:** Prevents category collision across features
6. **Combined Prompts:** Concatenate age + gender embeddings
7. **Dual Encoders:** Separate encoder/decoder prompt encoders (different parameters)
8. **Prompt Prepending:** Inject prompts before token sequences in encoder/decoder

**Key Files:**
- `conditional_prompt.py:10-65` - NumericalConditionalPrompt
- `conditional_prompt.py:68-138` - CategoricalConditionalPrompt
- `conditional_prompt.py:141-218` - ConditionalPrompt (combined)

---

## What's Next?

**Next:** [13_MULTI_TASK_LEARNING.md](13_MULTI_TASK_LEARNING.md) - Age/sex prediction heads, auxiliary losses, loss weight balancing

**Alternative:**
- [14_LOSS_FUNCTIONS.md](14_LOSS_FUNCTIONS.md) - Skip to loss computation
- [15_TRAINING_LOOP.md](15_TRAINING_LOOP.md) - Jump to training loop

---

**Navigation:**
- ← Back to [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md)
- → Next: [13_MULTI_TASK_LEARNING.md](13_MULTI_TASK_LEARNING.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
