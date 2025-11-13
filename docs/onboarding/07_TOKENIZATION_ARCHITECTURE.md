# 07: Tokenization Architecture - From Codes to Token IDs

**Estimated Time:** 75 minutes
**Prerequisites:** [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md)
**Next:** [08_DATA_LOADING.md](08_DATA_LOADING.md)

---

## Learning Objectives

- Understand the seven special tokens and their purposes
- Learn the token ID layout (0-6 special, 7+ codes)
- Understand code_offset = 7 and why it matters
- Master token sequence format: `<BOS> <v> code1 code2 <\v> <EOS>`
- Distinguish between vocabulary index and token ID
- Understand encoding (code → token ID) and decoding (token ID → code)

---

## The Token ID Problem

### Vocabulary Index vs Token ID

**DiagnosisVocabulary (previous page):**
```python
vocab.code2idx["401.9"] = 0  # Vocabulary index
vocab.code2idx["250.00"] = 1
vocab.code2idx["428.0"] = 2
```

**Problem:** BART needs special tokens (padding, BOS, EOS, etc.)

**Solution:** Reserve token IDs 0-6 for special tokens, shift code indices by +7

**DiagnosisCodeTokenizer (this page):**
```python
# Token ID layout:
0-6:  Special tokens (<PAD>, <BOS>, <EOS>, <v>, <\v>, <END>, <mask>)
7+:   Diagnosis codes (401.9 → 7, 250.00 → 8, 428.0 → 9)
```

**Mapping:**
```
Vocabulary Index  →  Token ID
----------------     ---------
     (N/A)       →     0-6     (special tokens)
       0         →      7      ("401.9" + offset 7)
       1         →      8      ("250.00" + offset 7)
       2         →      9      ("428.0" + offset 7)
     6984        →    6991     (last code + offset 7)
```

---

## The Seven Special Tokens

### Token Definitions (code_tokenizer.py:14-20, 31-39)

```python
special_token_ids = {
    "<PAD>": 0,   # Padding token (for batch processing)
    "<BOS>": 1,   # Beginning of sequence (BART <s>)
    "<EOS>": 2,   # End of sequence (BART </s>)
    "<v>": 3,     # Visit start marker
    "<\\v>": 4,   # Visit end marker
    "<END>": 5,   # Patient sequence end (legacy, rarely used)
    "<mask>": 6   # Mask token (for denoising)
}
```

### Special Token Purposes

| Token ID | Token | BART Equivalent | Purpose |
|----------|-------|-----------------|---------|
| 0 | `<PAD>` | `<pad>` | Pad sequences to same length in batch |
| 1 | `<BOS>` | `<s>` | Mark beginning of sequence |
| 2 | `<EOS>` | `</s>` | Mark end of sequence |
| 3 | `<v>` | (custom) | Mark start of hospital visit |
| 4 | `<\v>` | (custom) | Mark end of hospital visit |
| 5 | `<END>` | (custom) | Legacy end marker (not used) |
| 6 | `<mask>` | `<mask>` | Masked token for denoising |

**Key Insight:** Tokens 0-2 match BART conventions, tokens 3-6 are custom for EHR structure.

---

## Token Sequence Format

### Basic Format

**Single visit:**
```
<BOS> <v> code1 code2 code3 <\v> <EOS>
  1    3    7     8     9    4     2     (token IDs)
```

**Multiple visits:**
```
<BOS> <v> code1 code2 <\v> <v> code3 code4 <\v> <EOS>
  1    3    7     8    4    3   9    10   4     2
```

**Example with real codes:**
```
Patient: 65yo male, 2 visits
  Visit 1: ["401.9", "250.00"]  (hypertension, diabetes)
  Visit 2: ["428.0"]            (heart failure)

Token sequence:
<BOS> <v> 401.9 250.00 <\v> <v> 428.0 <\v> <EOS>
  1    3    7     8     4    3    9    4     2
```

### Visit Markers: Why `<v>` and `<\v>`?

**Purpose:**
- Preserve visit boundaries in flat sequence
- Enable model to learn visit-level patterns
- Support next-visit prediction (temporal perplexity metric)

**Alternative (NOT USED):** List of lists (harder for BART sequence model)

---

## The code_offset Concept

### Implementation (code_tokenizer.py:43-44)

```python
# Offset for medical codes (start after special tokens)
self.code_offset = len(self.special_token_ids)  # = 7
```

### Encoding with Offset (code_tokenizer.py:51-71)

```python
def encode_codes(self, codes: List[str]) -> List[int]:
    """Encode medical codes to token IDs."""
    token_ids = []
    for code in codes:
        if code in self.special_token_ids:
            token_ids.append(self.special_token_ids[code])  # Special token (0-6)
        else:
            vocab_idx = self.vocab.code2idx[code]  # Vocabulary index
            token_ids.append(vocab_idx + self.code_offset)  # Add offset (+7)
    return token_ids
```

**Example:**
```python
vocab.code2idx["401.9"] = 0  # Vocabulary index

tokenizer.encode_codes(["401.9"])
# vocab_idx = 0
# token_id = 0 + 7 = 7
# Returns: [7]
```

### Decoding with Offset (code_tokenizer.py:73-84)

```python
def decode_codes(self, token_ids: List[int]) -> List[str]:
    """Decode token IDs back to medical codes."""
    # Subtract offset to get vocabulary indices
    vocab_ids = [idx - self.code_offset for idx in token_ids if idx >= self.code_offset]
    return self.vocab.decode(vocab_ids)
```

**Example:**
```python
tokenizer.decode_codes([7, 8, 9])
# Filter: [7, 8, 9] (all >= 7)
# Subtract offset: [7-7, 8-7, 9-7] = [0, 1, 2]
# Decode: ["401.9", "250.00", "428.0"]
```

**Why `if idx >= self.code_offset`?**
- Filters out special tokens (0-6)
- Only decodes actual diagnosis codes
- Special tokens handled separately

---

## Encoding and Decoding Operations

### encode_codes() - Basic Encoding

**Input:** List of code strings
**Output:** List of token IDs

```python
tokenizer.encode_codes(["401.9", "250.00", "<mask>", "428.0"])
# Returns: [7, 8, 6, 9]
#          401.9=7, 250.00=8, <mask>=6, 428.0=9
```

### encode_visit() - Visit with Markers

**Input:** List of codes (single visit)
**Output:** Token IDs with `<v>` and `<\v>` markers

```python
tokenizer.encode_visit(["401.9", "250.00"], add_markers=True)
# Returns: [3, 7, 8, 4]
#          <v>=3, 401.9=7, 250.00=8, <\v>=4
```

### encode_patient() - Full Patient Sequence

[IN PROGRESS - See code_tokenizer.py:110-140 for implementation]

**Key points:**
- Adds `<BOS>` at start, `<EOS>` at end
- Wraps each visit with `<v>` and `<\v>`
- Concatenates all visits into single sequence

---

## Vocabulary Size vs Token Vocabulary Size

### Total Token Vocabulary

**Formula:**
```
total_vocab_size = num_special_tokens + num_diagnosis_codes
                 = 7 + 6,985
                 = 6,992 tokens
```

**BART embedding table:**
```python
nn.Embedding(num_embeddings=6992, embedding_dim=768)
# 6,992 token IDs × 768 dimensions = 5.4M parameters
```

### Comparison with BART-base

| Model | Special Tokens | Code/Word Tokens | Total Vocab |
|-------|----------------|------------------|-------------|
| BART-base | 4 | 50,261 | 50,265 |
| PromptEHR (ours) | 7 | 6,985 | 6,992 |

**PromptEHR has 7× smaller vocabulary than BART!**

---

## Try It Yourself

### Exercise 1: Token ID Mapping

```python
from vocabulary import DiagnosisVocabulary
from code_tokenizer import DiagnosisCodeTokenizer

# Build vocabulary
vocab = DiagnosisVocabulary()
vocab.add_codes(["401.9", "250.00", "428.0"])

# Create tokenizer
tokenizer = DiagnosisCodeTokenizer(vocab)

# Test token ID mapping
print(f"code_offset: {tokenizer.code_offset}")
print(f"\nVocabulary indices:")
print(f"  '401.9' → {vocab.code2idx['401.9']}")
print(f"  '250.00' → {vocab.code2idx['250.00']}")
print(f"  '428.0' → {vocab.code2idx['428.0']}")

print(f"\nToken IDs (with offset):")
token_ids = tokenizer.encode_codes(["401.9", "250.00", "428.0"])
print(f"  {token_ids}")
```

**Expected Output:**
```
code_offset: 7

Vocabulary indices:
  '401.9' → 0
  '250.00' → 1
  '428.0' → 2

Token IDs (with offset):
  [7, 8, 9]
```

### Exercise 2: Visit Encoding

```python
# Encode single visit
visit_codes = ["401.9", "250.00"]

# Without markers
no_markers = tokenizer.encode_visit(visit_codes, add_markers=False)
print(f"Without markers: {no_markers}")

# With markers
with_markers = tokenizer.encode_visit(visit_codes, add_markers=True)
print(f"With markers: {with_markers}")

# Decode back
decoded = tokenizer.decode_codes(with_markers)
print(f"Decoded: {decoded}")
```

**Expected Output:**
```
Without markers: [7, 8]
With markers: [3, 7, 8, 4]
Decoded: ['401.9', '250.00']  # Note: markers filtered out
```

[IN PROGRESS - Additional exercises on patient encoding, special token handling, and padding]

---

## Common Pitfalls

### Pitfall 1: Confusing Vocabulary Index with Token ID

**Wrong:**
```python
vocab_idx = vocab.code2idx["401.9"]  # = 0
# Using vocab_idx directly as token ID (WRONG!)
```

**Right:**
```python
token_id = tokenizer.encode_codes(["401.9"])[0]  # = 7 (with offset)
```

### Pitfall 2: Forgetting to Filter Special Tokens in Decoding

**Wrong:**
```python
token_ids = [1, 3, 7, 8, 4, 2]  # Includes <BOS>, <v>, codes, <\v>, <EOS>
vocab_ids = [id - 7 for id in token_ids]  # = [-6, -4, 0, 1, -3, -5]
vocab.decode(vocab_ids)  # KeyError: -6!
```

**Right:**
```python
# decode_codes() filters: if idx >= self.code_offset
tokenizer.decode_codes(token_ids)  # Only decodes [7, 8] → ["401.9", "250.00"]
```

[IN PROGRESS - Additional pitfalls]

---

## Key Design Decisions

### Decision 1: Fixed code_offset = 7

**Why 7?**
- Exactly matches number of special tokens
- Simple arithmetic: token_id = vocab_idx + 7
- Future-proof: Can add special tokens by increasing offset

### Decision 2: Custom Visit Markers (`<v>`, `<\v>`)

**Why not use BART's built-in tokens?**
- BART has no concept of "visits" (clinical structure)
- Custom tokens enable visit-level operations
- Model learns visit boundaries explicitly

[IN PROGRESS - Additional design decisions]

---

## Connection to Model Architecture

### Token IDs → Embeddings

```
Token ID 7 ("401.9")
    ↓
Embedding layer: embedding_table[7]
    ↓
768-dimensional vector
    ↓
BART encoder
```

**See:** [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) for embedding layer details

### Special Tokens in Training

**Padding (`<PAD>`, ID=0):**
- Labels set to -100 to ignore in loss computation
- Ensures variable-length sequences can batch

**Masking (`<mask>`, ID=6):**
- Used in data corruption for denoising objective
- Model learns to predict masked codes

**See:** [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md) for masking details

---

## Summary

**DiagnosisCodeTokenizer bridges vocabulary and BART:**

1. **Seven Special Tokens:** IDs 0-6 for sequence structure
2. **code_offset = 7:** Shifts diagnosis code IDs to avoid collision
3. **Token ID Layout:** 0-6 (special) + 7-6991 (codes) = 6,992 total
4. **Visit Markers:** `<v>` and `<\v>` preserve visit boundaries
5. **Encoding:** vocab_idx + 7 → token_id
6. **Decoding:** token_id - 7 → vocab_idx (filter special tokens)
7. **Total Vocabulary:** 6,992 tokens (7× smaller than BART)

**Key Files:**
- `code_tokenizer.py:10-100` - DiagnosisCodeTokenizer class
- `code_tokenizer.py:31-39` - Special token definitions
- `code_tokenizer.py:43-44` - code_offset = 7

---

## What's Next?

**Next:** [08_DATA_LOADING.md](08_DATA_LOADING.md) - Deep dive into load_mimic_data() function and data preprocessing pipeline

**Alternative:**
- [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md) - Skip to data corruption (denoising objective)
- [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) - Jump to model architecture

---

**Navigation:**
- ← Back to [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md)
- → Next: [08_DATA_LOADING.md](08_DATA_LOADING.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
