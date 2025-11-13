# 09: Dataset Corruption - BART-Style Denoising for EHR Generation

**Estimated Time:** 90 minutes
**Prerequisites:** [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)
**Next:** [10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md)

---

## Learning Objectives

- Understand BART's denoising objective and why it matters
- Master three corruption strategies: mask infilling, token deletion, token replacement
- Learn Poisson(λ=3) span masking for realistic corruption
- Understand label masking (padding → -100) for loss calculation
- Learn corruption probability = 0.5 (half corrupted, half clean)
- Understand EHRPatientDataset and EHRDataCollator classes

---

## The Denoising Objective

### Why Corrupt the Data?

**Problem:** Training autoregressive models on clean data leads to:
- Overfitting to exact code sequences
- Poor generalization to new patterns
- No robustness to noise or missing data

**Solution:** BART-style denoising
- Corrupt input sequences (encoder sees corrupted)
- Keep target sequences clean (decoder predicts original)
- Model learns to reconstruct from noisy input
- Forces model to understand code relationships, not just memorize

### BART Denoising Framework

**Original BART (text):**
```
Clean text:      "The cat sat on the mat"
Corrupted input: "The <mask> on the mat"
Target output:   "The cat sat on the mat"
```

**PromptEHR (diagnosis codes):**
```
Clean sequence:  <v> 401.9 250.00 585.9 <\v>
Corrupted input: <v> 401.9 <mask> <\v>
Target output:   <v> 401.9 250.00 585.9 <\v>
```

**Key insight:** Encoder sees corrupted, decoder predicts clean.

---

## Three Corruption Strategies

### Overview (dataset.py:14-44)

```python
class CorruptionFunctions:
    """Data corruption functions for robust EHR generation training.

    Implements three corruption strategies:
    1. Mask infilling: Replace code spans with <mask> token
    2. Token deletion: Randomly delete codes
    3. Token replacement: Replace codes with random alternatives
    """

    def __init__(
        self,
        tokenizer: DiagnosisCodeTokenizer,
        lambda_poisson: float = 3.0,      # Span length distribution
        del_probability: float = 0.15,    # Deletion probability
        rep_probability: float = 0.15     # Replacement probability
    ):
```

**Three strategies applied with equal probability (1/3 each):**
- 33% chance: Mask infilling
- 33% chance: Token deletion
- 33% chance: Token replacement

---

## Strategy 1: Mask Infilling (Span Masking)

### Algorithm (dataset.py:45-106)

```python
def mask_infill(
    self,
    visits: List[List[str]]
) -> Tuple[List[List[str]], List[List[int]]]:
    """Apply Poisson-distributed span masking to diagnosis codes."""
```

**Steps for each visit:**
1. Sample span length from Poisson(λ=3.0)
2. Randomly select contiguous span of that length
3. Replace entire span with single `<mask>` token
4. Track which positions were masked (for loss calculation)

### Poisson Distribution (λ=3.0)

**Why Poisson?**
- Models natural variation in missing code counts
- λ=3.0 → Average span length = 3 codes
- Prevents uniform corruption (too predictable)

**Span length distribution:**
```
P(length=1) = 5%   → Single code masked
P(length=2) = 22%  → Two codes masked
P(length=3) = 22%  → Three codes masked (most common)
P(length=4) = 17%
P(length=5+) = 34% → Long spans
```

### Example

**Input visit:**
```python
visit = ["401.9", "250.00", "585.9", "428.0", "584.9"]
```

**Corruption process:**
1. Sample span length: `np.random.poisson(3.0)` → 3
2. Select start position: `start_idx = 1` (random)
3. Masked span: `visit[1:4]` = `["250.00", "585.9", "428.0"]`
4. Replace with `<mask>`

**Output:**
```python
corrupted_visit = ["401.9", "<mask>", "584.9"]
label_mask = [0, 1, 1, 1, 0]  # 1 = predict this position
```

**Label mask interpretation:**
- 0: Don't compute loss for this position (already in input)
- 1: Compute loss for this position (masked, model must predict)

### Implementation (dataset.py:83-106)

```python
# Sample span length from Poisson distribution
span_length = max(1, min(num_codes - 1,
                        np.random.poisson(self.lambda_poisson)))

# Randomly select start position
max_start = num_codes - span_length
start_idx = np.random.randint(0, max(1, max_start + 1))

# Create corrupted visit
corrupted_visit = (
    visit[:start_idx] +          # Keep codes before span
    [self.mask_token] +           # Single <mask> token
    visit[start_idx + span_length:]  # Keep codes after span
)

# Create label mask (1 for masked positions)
label_mask = [0] * num_codes
for i in range(start_idx, min(start_idx + span_length, num_codes)):
    label_mask[i] = 1
```

**Key details:**
- `max(1, ...)` → Minimum span length = 1
- `min(num_codes - 1, ...)` → Never mask entire visit
- Single `<mask>` token replaces entire span (span infilling)

---

## Strategy 2: Token Deletion

### Algorithm (dataset.py:108-152)

```python
def del_token(self, visits: List[List[str]]) -> List[List[str]]:
    """Apply binomial token deletion to diagnosis codes.

    For each code in each visit:
    - With probability del_probability (0.15), delete the code
    - Never delete all codes in a visit (keep at least 1)
    """
```

**Steps:**
1. For each code: Flip biased coin (p=0.15)
2. If heads: Delete code
3. If all codes would be deleted: Keep random code (safety)

### Example

**Input visit:**
```python
visit = ["401.9", "250.00", "585.9", "428.0", "584.9"]
```

**Deletion process:**
```python
# Generate deletion mask (1 = delete, 0 = keep)
deletion_mask = [0, 1, 0, 1, 0]  # Random with p=0.15

# Result: Delete "250.00" and "428.0"
corrupted_visit = ["401.9", "585.9", "584.9"]
```

### Implementation (dataset.py:134-149)

```python
# Generate deletion mask (1 = delete, 0 = keep)
deletion_mask = np.random.binomial(1, self.del_probability, num_codes)

# Keep at least 1 code per visit
if deletion_mask.sum() == num_codes:
    # All would be deleted, randomly keep one
    keep_idx = np.random.randint(0, num_codes)
    deletion_mask[keep_idx] = 0

# Apply deletion
corrupted_visit = [
    code for i, code in enumerate(visit)
    if deletion_mask[i] == 0
]
```

**Why binomial (not Poisson)?**
- Independent deletion decision per code
- Simulates missing diagnoses (real-world scenario)
- No fixed number of deletions (unlike span masking)

---

## Strategy 3: Token Replacement

### Algorithm (dataset.py:154-200)

```python
def rep_token(self, visits: List[List[str]]) -> List[List[str]]:
    """Apply binomial token replacement to diagnosis codes.

    For each code in each visit:
    - With probability rep_probability (0.15), replace with random code
    - Random code sampled uniformly from vocabulary
    """
```

**Steps:**
1. For each code: Flip biased coin (p=0.15)
2. If heads: Replace with random code from vocabulary
3. Random code selection: Uniform distribution over all 6,985 codes

### Example

**Input visit:**
```python
visit = ["401.9", "250.00", "585.9"]
```

**Replacement process:**
```python
# Generate replacement mask
replacement_mask = [0, 1, 0]  # Random with p=0.15

# Replace "250.00" with random code (e.g., "486")
corrupted_visit = ["401.9", "486", "585.9"]
```

**Why random replacement?**
- Simulates coding errors (wrong ICD-9 code assigned)
- Forces model to learn code coherence (detect implausible codes)
- Harder than deletion (model sees wrong code, not just missing)

### Implementation

```python
# Generate replacement mask
replacement_mask = np.random.binomial(1, self.rep_probability, num_codes)

# Sample random codes from vocabulary
random_codes = [
    self.tokenizer.vocab.idx2code[np.random.randint(0, self.vocab_size)]
    for _ in range(replacement_mask.sum())
]

# Apply replacement
corrupted_visit = []
random_idx = 0
for i, code in enumerate(visit):
    if replacement_mask[i] == 1:
        corrupted_visit.append(random_codes[random_idx])
        random_idx += 1
    else:
        corrupted_visit.append(code)
```

---

## Corruption Probability = 0.5

### Why 50% Corruption Rate?

**Implementation (dataset.py:250-270):**
```python
def corrupt_sequence(
    self,
    visits: List[List[str]],
    corruption_prob: float = 0.5
) -> Tuple[List[List[str]], Optional[List[List[int]]]]:
    """Corrupt patient visit sequence with 50% probability."""

    # Flip coin: corrupt or keep clean
    if np.random.random() < corruption_prob:
        # Apply one of three corruption strategies (uniform random)
        strategy = np.random.randint(0, 3)
        if strategy == 0:
            return self.mask_infill(visits)
        elif strategy == 1:
            return self.del_token(visits), None
        else:
            return self.rep_token(visits), None
    else:
        # No corruption (clean sequence)
        return visits, None
```

**Corruption breakdown:**
- 50% chance: No corruption (clean)
- 50% chance: Corrupted
  - 16.7% (1/3 of 50%): Mask infilling
  - 16.7%: Token deletion
  - 16.7%: Token replacement

**Why 50/50 split?**
- Balance between reconstruction and generation
- Model learns both:
  - **Denoising:** Reconstruct from corrupted input (50%)
  - **Autoregressive:** Generate from clean input (50%)
- Too much corruption (>70%): Model struggles to learn
- Too little corruption (<30%): Overfitting to clean sequences

---

## Label Masking for Loss Calculation

### The Padding Problem

**PyTorch cross-entropy loss computes:**
```python
loss = CrossEntropyLoss(predictions, labels)
```

**Problem:** Padding tokens (`<PAD>`, ID=0) should not contribute to loss

**Solution:** Set padding labels to -100 (PyTorch ignores -100 in loss)

### Implementation (dataset.py:300-350)

```python
def __getitem__(self, idx: int) -> Dict:
    """Get patient sample with corruption and tokenization."""

    # ... tokenize sequence ...

    # Create labels (decoder target)
    labels = input_ids.clone()

    # CRITICAL: Mask padding tokens in labels
    labels[labels == self.tokenizer.pad_token_id] = -100

    return {
        'input_ids': input_ids,      # Encoder input (may be corrupted)
        'labels': labels,             # Decoder target (original sequence)
        'x_num': x_num,               # Age (continuous conditioning)
        'x_cat': x_cat                # Gender (categorical conditioning)
    }
```

**Why -100?**
- PyTorch convention: `ignore_index=-100` in `CrossEntropyLoss`
- Loss computation skips positions with label=-100
- Prevents model from learning to predict padding

### Label Masking for Span Masking

**Mask infilling also uses label masking:**

```python
# Only compute loss for masked positions
label_mask = [0, 1, 1, 1, 0]  # 1 = compute loss

# Convert to -100 for padding
labels = [code_id if mask == 1 else -100
          for code_id, mask in zip(labels, label_mask)]
```

**Result:** Loss only computed for positions that were masked.

---

## EHRPatientDataset Class

### Structure (dataset.py:200-350)

```python
class EHRPatientDataset(Dataset):
    """PyTorch dataset for EHR patient sequences with corruption."""

    def __init__(
        self,
        patient_records: List[PatientRecord],
        tokenizer: DiagnosisCodeTokenizer,
        max_seq_len: int = 512,
        corruption_prob: float = 0.5,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15
    ):
        self.patient_records = patient_records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Initialize corruption functions
        self.corruptor = CorruptionFunctions(
            tokenizer=tokenizer,
            lambda_poisson=lambda_poisson,
            del_probability=del_probability,
            rep_probability=rep_probability
        )
        self.corruption_prob = corruption_prob
```

**Key responsibilities:**
1. Load PatientRecord objects
2. Apply corruption (50% probability)
3. Tokenize sequences (codes → token IDs)
4. Pad to max_seq_len (batch processing)
5. Create labels with -100 for padding
6. Return dictionary with input_ids, labels, demographics

---

## EHRDataCollator Class

### Purpose

**Problem:** Patients have different sequence lengths
- Patient A: 20 tokens
- Patient B: 150 tokens
- Patient C: 80 tokens

**Solution:** Pad all sequences to same length in batch

### Implementation (dataset.py:400-450)

```python
class EHRDataCollator:
    """Collator for batching EHR patient sequences with padding."""

    def __init__(self, tokenizer: DiagnosisCodeTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with padding."""

        # Find max length in batch
        max_len = max(len(item['input_ids']) for item in batch)

        # Pad all sequences to max_len
        input_ids = []
        labels = []
        for item in batch:
            # Pad input_ids with pad_token_id (0)
            padded_input = torch.cat([
                item['input_ids'],
                torch.full((max_len - len(item['input_ids']),), self.pad_token_id)
            ])
            input_ids.append(padded_input)

            # Pad labels with -100 (ignored in loss)
            padded_labels = torch.cat([
                item['labels'],
                torch.full((max_len - len(item['labels']),), -100)
            ])
            labels.append(padded_labels)

        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'x_num': torch.stack([item['x_num'] for item in batch]),
            'x_cat': torch.stack([item['x_cat'] for item in batch])
        }
```

**Key operations:**
- Find max length in batch (dynamic per batch)
- Pad shorter sequences to max length
- Input padding: Use `<PAD>` token (ID=0)
- Label padding: Use -100 (ignored in loss)

---

## Example: Full Corruption Pipeline

### Input Patient

```python
PatientRecord(
    subject_id=10006,
    age=65.0,
    gender='M',
    visits=[
        ["401.9", "250.00", "585.9"],  # Visit 1
        ["428.0", "584.9"]              # Visit 2
    ]
)
```

### Step 1: Corruption (Mask Infilling)

**Original visits:**
```python
[["401.9", "250.00", "585.9"], ["428.0", "584.9"]]
```

**After corruption (span masking in visit 1):**
```python
[["401.9", "<mask>"], ["428.0", "584.9"]]
# Masked: "250.00", "585.9"
```

### Step 2: Tokenization

**Corrupted sequence:**
```python
input_ids = [1, 3, 7, 6, 4, 3, 9, 10, 4, 2]
# 1=<BOS>, 3=<v>, 7=401.9, 6=<mask>, 4=<\v>, 3=<v>, 9=428.0, 10=584.9, 4=<\v>, 2=<EOS>
```

**Target sequence (original, uncorrupted):**
```python
labels = [1, 3, 7, 8, 11, 4, 3, 9, 10, 4, 2]
# 1=<BOS>, 3=<v>, 7=401.9, 8=250.00, 11=585.9, 4=<\v>, 3=<v>, 9=428.0, 10=584.9, 4=<\v>, 2=<EOS>
```

### Step 3: Label Masking

**Mask non-content tokens:**
```python
# Set special tokens and uncorrupted codes to -100
labels = [-100, -100, -100, 8, 11, -100, -100, -100, -100, -100, -100]
# Only compute loss for positions 3-4 (the masked span)
```

### Step 4: Batch with Padding

**Batch of 3 patients (lengths: 10, 15, 8):**
```python
# Pad all to max_len=15
input_ids = torch.tensor([
    [1, 3, 7, 6, 4, 3, 9, 10, 4, 2, 0, 0, 0, 0, 0],   # Patient 1 (padded)
    [1, 3, 5, 12, 13, 4, 3, 14, 15, 16, 4, 3, 17, 4, 2],  # Patient 2 (full)
    [1, 3, 18, 19, 4, 3, 20, 2, 0, 0, 0, 0, 0, 0, 0]   # Patient 3 (padded)
])
```

---

## Try It Yourself

### Exercise 1: Test Span Masking

```python
from dataset import CorruptionFunctions
from code_tokenizer import DiagnosisCodeTokenizer
from vocabulary import DiagnosisVocabulary

# Setup
vocab = DiagnosisVocabulary()
vocab.add_codes(["401.9", "250.00", "585.9", "428.0", "584.9"])
tokenizer = DiagnosisCodeTokenizer(vocab)

corruptor = CorruptionFunctions(tokenizer, lambda_poisson=3.0)

# Test mask infilling
visits = [["401.9", "250.00", "585.9", "428.0", "584.9"]]
corrupted, label_mask = corruptor.mask_infill(visits)

print(f"Original:  {visits[0]}")
print(f"Corrupted: {corrupted[0]}")
print(f"Label mask: {label_mask[0]}")
```

**Expected output (random):**
```
Original:  ['401.9', '250.00', '585.9', '428.0', '584.9']
Corrupted: ['401.9', '<mask>', '584.9']
Label mask: [0, 1, 1, 1, 0]
```

[IN PROGRESS - Additional exercises on deletion, replacement, and full pipeline]

---

## Summary

**Dataset corruption is critical for robust EHR generation:**

1. **Three Strategies:** Mask infilling (Poisson λ=3), token deletion (p=0.15), token replacement (p=0.15)
2. **Corruption Rate:** 50% corrupted, 50% clean (balance denoising and generation)
3. **Span Masking:** Poisson-distributed span lengths (average 3 codes)
4. **Label Masking:** Padding → -100 (ignored in PyTorch loss)
5. **EHRPatientDataset:** Applies corruption, tokenization, and padding
6. **EHRDataCollator:** Batches sequences with dynamic padding
7. **Denoising Objective:** Encoder sees corrupted, decoder predicts clean

**Key Files:**
- `dataset.py:14-106` - Mask infilling implementation
- `dataset.py:108-200` - Token deletion and replacement
- `dataset.py:200-350` - EHRPatientDataset class
- `dataset.py:400-450` - EHRDataCollator class

---

## What's Next?

**Next:** [10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md) - End-to-end data pipeline from CSV to model input

**Skip to ML architecture:**
- [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) - BART encoder-decoder architecture
- [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md) - Demographic conditioning

---

**Navigation:**
- ← Back to [08_DATA_LOADING.md](08_DATA_LOADING.md)
- → Next: [10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
