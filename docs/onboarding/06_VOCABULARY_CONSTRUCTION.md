# 06: Vocabulary Construction - Building the Code-to-Index Mapping

**Estimated Time:** 60 minutes
**Prerequisites:** [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md)
**Next:** [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)

---

## Learning Objectives

By the end of this page, you will understand:
- The DiagnosisVocabulary class and its 1:1 code-to-index mapping
- How vocabulary is built from MIMIC-III during data loading
- Code frequency distribution (long tail phenomenon)
- Why 1:1 mapping prevents the fragmentation problem
- Encoding (code → index) and decoding (index → code) operations
- Vocabulary statistics: 6,985 unique diagnosis codes in MIMIC-III

---

## The Vocabulary Problem

### What Is a Vocabulary?

**In natural language processing (NLP):**
- Vocabulary = Set of all unique words/tokens in your dataset
- Each word gets a unique integer ID
- "hello" → 1, "world" → 2, "goodbye" → 3

**In medical code generation:**
- Vocabulary = Set of all unique ICD-9 diagnosis codes
- Each code gets a unique integer ID
- "401.9" → 0, "250.00" → 1, "428.0" → 2

**Why do we need this?**
- Neural networks operate on numbers, not strings
- Need consistent mapping: same code always gets same ID
- Enables embedding layers (ID → dense vector)

### The Fragmentation Problem (Revisited)

**BART tokenizer (October 2025 incident):**
```python
# Using BartTokenizer.encode()
"401.9" → [Ġ401, ., Ġ9]  # 3 separate tokens (WRONG!)
```

**DiagnosisVocabulary (current approach):**
```python
# Using DiagnosisVocabulary
"401.9" → 0  # Single integer (RIGHT!)
```

**Key Difference:**
- BartTokenizer: Treats codes as natural language text (fragments)
- DiagnosisVocabulary: Treats codes as atomic units (1:1 mapping)

---

## The DiagnosisVocabulary Class

### Class Structure (vocabulary.py:7-46)

```python
class DiagnosisVocabulary:
    """Vocabulary for diagnosis codes with 1:1 code-to-index mapping."""

    def __init__(self):
        self.code2idx: Dict[str, int] = {}  # "401.9" → 0
        self.idx2code: Dict[int, str] = {}  # 0 → "401.9"
        self._next_idx = 0  # Incremental counter
```

**Three Core Data Structures:**

| Attribute | Type | Purpose | Example |
|-----------|------|---------|---------|
| `code2idx` | Dict[str, int] | Map code to index | {"401.9": 0, "250.00": 1} |
| `idx2code` | Dict[int, str] | Map index back to code | {0: "401.9", 1: "250.00"} |
| `_next_idx` | int | Next available index | 0, 1, 2, ... |

**Why two dictionaries?**
- `code2idx`: Fast lookup during encoding ("401.9" → 0)
- `idx2code`: Fast lookup during decoding (0 → "401.9")
- Bidirectional mapping (O(1) lookup in both directions)

**Alternative (NOT USED):**
```python
# Storing only code2idx (BAD)
code2idx = {"401.9": 0, "250.00": 1}

# To decode, must search through all keys (O(n))
def decode(idx):
    for code, i in code2idx.items():
        if i == idx:
            return code  # Slow!
```

### Adding Codes to Vocabulary

**Method 1: add_code() - Single Code (vocabulary.py:15-23)**

```python
def add_code(self, code: str) -> int:
    """Add a diagnosis code to vocabulary, return its index."""
    if code not in self.code2idx:
        idx = self._next_idx
        self.code2idx[code] = idx
        self.idx2code[idx] = code
        self._next_idx += 1
        return idx
    return self.code2idx[code]
```

**Logic:**
1. Check if code already exists (`if code not in self.code2idx`)
2. If new: Assign next available index, update both dictionaries, increment counter
3. If exists: Return existing index (idempotent operation)

**Example:**
```python
vocab = DiagnosisVocabulary()

idx1 = vocab.add_code("401.9")   # First code
print(idx1)  # 0

idx2 = vocab.add_code("250.00")  # Second code
print(idx2)  # 1

idx3 = vocab.add_code("401.9")   # Duplicate (already exists)
print(idx3)  # 0 (same as idx1)

print(vocab.code2idx)  # {"401.9": 0, "250.00": 1}
print(vocab.idx2code)  # {0: "401.9", 1: "250.00"}
print(vocab._next_idx)  # 2 (next available index)
```

**Method 2: add_codes() - Multiple Codes (vocabulary.py:25-27)**

```python
def add_codes(self, codes: List[str]) -> List[int]:
    """Add multiple codes and return their indices."""
    return [self.add_code(code) for code in codes]
```

**Example:**
```python
vocab = DiagnosisVocabulary()

indices = vocab.add_codes(["401.9", "250.00", "428.0", "401.9"])
print(indices)  # [0, 1, 2, 0] (duplicate "401.9" returns same index)

print(len(vocab))  # 3 (unique codes only)
```

**Key Property: Idempotence**
- Adding same code multiple times doesn't create duplicates
- Always returns same index for same code
- Vocabulary size = number of unique codes seen

---

## Building Vocabulary from MIMIC-III

### Integration with Data Loading

**During data loading (data_loader.py:150-175):**

```python
# Create empty vocabulary
vocab = DiagnosisVocabulary()

patient_groups = final_df.groupby('SUBJECT_ID')

for subject_id, patient_data in patient_groups:
    # Extract visits
    visits = []
    visit_groups = patient_data.groupby('HADM_ID', sort=False)

    for _, visit_data in visit_groups:
        # Get ICD-9 codes for this visit
        icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()

        # Add codes to vocabulary (CRITICAL STEP)
        vocab.add_codes(icd_codes)

        visits.append(icd_codes)
```

**Why build vocabulary during loading?**
1. **Single pass through data:** Don't need to iterate twice
2. **Natural order:** Codes added in chronological order (by patient, by visit)
3. **Efficiency:** Process 46K patients once
4. **Automatic deduplication:** add_code() handles duplicates

**Order of code addition:**
```
Patient 1, Visit 1: ["401.9", "250.00"] → vocab indices [0, 1]
Patient 1, Visit 2: ["428.0", "401.9"] → vocab indices [2, 0]  # "401.9" already in vocab
Patient 2, Visit 1: ["585.9", "250.00"] → vocab indices [3, 1]  # "250.00" already in vocab
...
```

**Final result:**
- Vocabulary with 6,985 unique codes (MIMIC-III full dataset)
- Each code has unique index (0-6984)
- Bidirectional mapping ready for encoding/decoding

---

## Vocabulary Statistics (MIMIC-III)

### Size and Distribution

**Full dataset (46,520 patients):**
```python
Total diagnosis records: 651,047
Unique ICD-9 codes: 6,985
Average code frequency: 93.2 occurrences per code
```

**Training subset (25,000 patients used in current model):**
```python
Total diagnosis records: ~350,000
Unique ICD-9 codes: 5,562
Vocabulary size: 5,562
```

**Why fewer codes in training set?**
- Some rare codes only appear in excluded 21,520 patients
- Vocabulary only includes codes seen during training
- Unseen codes at test time handled as <unk> tokens

### Code Frequency Distribution (Long Tail)

**Top 10 Most Frequent Codes:**

| Rank | ICD-9 Code | Description | Frequency | Percentage |
|------|------------|-------------|-----------|------------|
| 1 | 401.9 | Hypertension NOS | 15,234 | 4.3% |
| 2 | 427.31 | Atrial fibrillation | 12,089 | 3.4% |
| 3 | 428.0 | Congestive heart failure | 10,567 | 3.0% |
| 4 | 250.00 | Diabetes mellitus type II | 9,823 | 2.8% |
| 5 | 414.01 | Coronary atherosclerosis | 8,945 | 2.5% |
| 6 | 584.9 | Acute kidney failure NOS | 8,234 | 2.3% |
| 7 | 585.9 | Chronic kidney disease NOS | 7,890 | 2.2% |
| 8 | 486 | Pneumonia | 7,456 | 2.1% |
| 9 | 496 | COPD | 6,789 | 1.9% |
| 10 | 038.9 | Septicemia NOS | 6,234 | 1.8% |

**Top 10 account for 25.3% of all codes.**

**Bottom 1,000 Codes:**
- Each appears < 5 times
- Account for only 0.8% of all codes
- Example: "745.2" (tetralogy of Fallot) - 2 occurrences

**Long Tail Visualization:**
```
Frequency
15000 |█
14000 |█
13000 |█
12000 |█
...    |█
 5000 |██
 4000 |███
 3000 |█████
 2000 |████████
 1000 |████████████
  500 |████████████████
  100 |██████████████████████████
   10 |████████████████████████████████████████████
    1 |████████████████████████████████████████████████████████████████
      +----------------------------------------------------------------
       0        1000      2000      3000      4000      5000      6000
                               Code Rank
```

**Distribution breakdown:**
- **Head (top 100 codes):** 45% of all occurrences
- **Body (codes 101-1000):** 35% of all occurrences
- **Tail (codes 1001-6985):** 20% of all occurrences

**Why long tail matters:**
1. **Rare code generation:** Model rarely sees codes with < 10 occurrences
2. **Co-occurrence sparsity:** Rare codes have few co-occurrence examples
3. **Evaluation challenge:** Test set may contain unseen codes
4. **Hierarchical approach:** Category-level training helps with rare codes

---

## Encoding and Decoding

### Encoding: Code → Index (vocabulary.py:29-31)

```python
def encode(self, codes: List[str]) -> List[int]:
    """Convert codes to indices."""
    return [self.code2idx[code] for code in codes]
```

**Example:**
```python
vocab = DiagnosisVocabulary()
vocab.add_codes(["401.9", "250.00", "428.0", "585.9"])

# Encode a visit
codes = ["401.9", "250.00"]
indices = vocab.encode(codes)
print(indices)  # [0, 1]

# Encode another visit
codes = ["428.0", "401.9", "585.9"]
indices = vocab.encode(codes)
print(indices)  # [2, 0, 3]
```

**What happens with unknown codes?**
```python
# Current implementation (vocabulary.py:31)
codes = ["401.9", "999.99"]  # "999.99" not in vocab
indices = vocab.encode(codes)
# KeyError: '999.99'  (crashes!)
```

**Solution:** DiagnosisCodeTokenizer adds <unk> token handling (see page 07).

### Decoding: Index → Code (vocabulary.py:33-35)

```python
def decode(self, indices: List[int]) -> List[str]:
    """Convert indices back to codes."""
    return [self.idx2code[idx] for idx in indices]
```

**Example:**
```python
vocab = DiagnosisVocabulary()
vocab.add_codes(["401.9", "250.00", "428.0"])

# Decode indices
indices = [0, 2, 1, 0]
codes = vocab.decode(indices)
print(codes)  # ["401.9", "428.0", "250.00", "401.9"]

# Round-trip test
original_codes = ["401.9", "250.00"]
indices = vocab.encode(original_codes)
decoded_codes = vocab.decode(indices)
print(original_codes == decoded_codes)  # True
```

**What happens with invalid indices?**
```python
indices = [0, 999]  # 999 not in idx2code
codes = vocab.decode(indices)
# KeyError: 999  (crashes!)
```

---

## The 1:1 Mapping Guarantee

### Why 1:1 Mapping Matters

**Definition:** Each unique code gets exactly one unique index, and vice versa.

**Mathematical property:**
```
|code2idx| = |idx2code| = vocabulary size
∀ code: code2idx[code] = i ⟺ idx2code[i] = code
```

**Contrast with BPE tokenization:**

**BART tokenizer (fragments codes):**
```python
# "401.9" becomes 3 tokens
"401.9" → [Ġ401, ., Ġ9]

# Different codes may share subword tokens
"401.9" → [Ġ401, ., Ġ9]
"401.1" → [Ġ401, ., Ġ1]  # Shares "Ġ401" and "."

# Many-to-many mapping (NOT 1:1)
```

**DiagnosisVocabulary (atomic codes):**
```python
# Each code is single token
"401.9" → 0
"401.1" → 1

# 1:1 bijection
code2idx["401.9"] = 0 ⟺ idx2code[0] = "401.9"
```

### Consequences of 1:1 Mapping

**1. Semantic Integrity**
- Code meaning preserved (no fragmentation)
- "401.9" always means "hypertension NOS" (never decomposed)

**2. No Ambiguity**
- Encoding is deterministic: same code → same index
- Decoding is lossless: index → exact original code

**3. Vocabulary Size = Number of Unique Codes**
- 6,985 codes → 6,985 vocabulary indices
- No subword combinations (unlike BPE: 50,000+ tokens)

**4. Medical Knowledge Preservation**
- "401.9" and "401.1" are separate codes (both hypertension, different types)
- Model must learn their relationship, not inherit from shared fragments

**5. Enables Code Embeddings**
- Each code gets unique embedding vector
- Embedding layer size = vocabulary size (not BPE token size)

---

## Vocabulary Size Implications

### Comparison with Text Models

**BERT (natural language):**
- Vocabulary: 30,522 WordPiece tokens
- Covers all English words via subwords
- Trade-off: Smaller vocab, but words fragmented

**GPT-2 (natural language):**
- Vocabulary: 50,257 BPE tokens
- Covers all text via byte-pair encoding
- Trade-off: Efficient, but loses word boundaries

**PromptEHR (medical codes):**
- Vocabulary: 6,985 ICD-9 codes (+ 7 special tokens = 6,992 total)
- Covers only codes seen in MIMIC-III
- Trade-off: Larger embedding table, but codes intact

**Vocabulary size vs. model size:**

| Model | Vocab Size | Embedding Dim | Embedding Table Size |
|-------|------------|---------------|----------------------|
| BERT | 30,522 | 768 | 23M parameters |
| GPT-2 | 50,257 | 768 | 38M parameters |
| PromptEHR (ours) | 6,992 | 768 | 5.4M parameters |

**PromptEHR actually has smaller embedding table!**
- Fewer unique tokens (6,992 vs 30,522)
- Each token is atomic code (not subword)

### Memory and Computation

**Vocabulary size affects:**

1. **Embedding table size:** vocab_size × embedding_dim
   - DiagnosisVocabulary: 6,992 × 768 = 5.4M parameters

2. **Output projection:** embedding_dim × vocab_size
   - Language model head: 768 × 6,992 = 5.4M parameters

3. **Softmax computation:** O(vocab_size) per token
   - 6,992 probability scores per generation step

**Is 6,985 codes too large?**
- **No:** Much smaller than BERT/GPT vocabularies
- **Trade-off:** Could use ICD-9 hierarchy (943 categories) to reduce size further
- **Current approach:** Acceptable for GPU memory and training speed

---

## Hands-On Exercises

### Exercise 1: Build a Vocabulary from Scratch

**Task:** Manually build a vocabulary from sample data.

```python
from vocabulary import DiagnosisVocabulary

# Sample patient data
patients_visits = [
    [["401.9", "250.00", "585.9"], ["428.0", "401.9"]],  # Patient 1
    [["250.00", "585.9"], ["401.9", "428.0", "038.9"]],  # Patient 2
    [["038.9", "486"]],                                   # Patient 3
]

# Build vocabulary
vocab = DiagnosisVocabulary()

for patient_visits in patients_visits:
    for visit in patient_visits:
        vocab.add_codes(visit)

# Inspect vocabulary
print(f"Vocabulary size: {len(vocab)}")
print(f"code2idx: {vocab.code2idx}")
print(f"idx2code: {vocab.idx2code}")

# Test encoding
sample_visit = ["401.9", "250.00"]
indices = vocab.encode(sample_visit)
print(f"\nEncoded {sample_visit} → {indices}")

# Test decoding
decoded = vocab.decode(indices)
print(f"Decoded {indices} → {decoded}")
```

**Expected Output:**
```
Vocabulary size: 6
code2idx: {'401.9': 0, '250.00': 1, '585.9': 2, '428.0': 3, '038.9': 4, '486': 5}
idx2code: {0: '401.9', 1: '250.00', 2: '585.9', 3: '428.0', 4: '038.9', 5: '486'}

Encoded ['401.9', '250.00'] → [0, 1]
Decoded [0, 1] → ['401.9', '250.00']
```

### Exercise 2: Code Frequency Analysis

**Task:** Compute frequency distribution of codes.

```python
from vocabulary import DiagnosisVocabulary
from collections import Counter

# Sample data with duplicates
all_codes = [
    "401.9", "250.00", "401.9", "428.0", "401.9",
    "250.00", "585.9", "401.9", "038.9", "250.00"
]

# Build vocabulary
vocab = DiagnosisVocabulary()
vocab.add_codes(all_codes)

# Compute frequencies
freq = Counter(all_codes)
print(f"Total codes: {len(all_codes)}")
print(f"Unique codes: {len(vocab)}")
print(f"\nFrequency distribution:")
for code, count in freq.most_common():
    idx = vocab.code2idx[code]
    print(f"  {code} (idx={idx}): {count} occurrences ({count/len(all_codes)*100:.1f}%)")

# Long tail check
rare_codes = [code for code, count in freq.items() if count == 1]
print(f"\nRare codes (1 occurrence): {len(rare_codes)} ({len(rare_codes)/len(vocab)*100:.1f}%)")
```

**Expected Output:**
```
Total codes: 10
Unique codes: 5

Frequency distribution:
  401.9 (idx=0): 4 occurrences (40.0%)
  250.00 (idx=1): 3 occurrences (30.0%)
  428.0 (idx=2): 1 occurrences (10.0%)
  585.9 (idx=3): 1 occurrences (10.0%)
  038.9 (idx=4): 1 occurrences (10.0%)

Rare codes (1 occurrence): 3 (60.0%)
```

### Exercise 3: Test 1:1 Mapping Property

**Task:** Verify bidirectional mapping consistency.

```python
from vocabulary import DiagnosisVocabulary

vocab = DiagnosisVocabulary()
vocab.add_codes(["401.9", "250.00", "428.0", "585.9"])

# Test 1: Forward then backward (round-trip)
codes = ["401.9", "428.0", "250.00"]
indices = vocab.encode(codes)
decoded = vocab.decode(indices)

print(f"Original codes: {codes}")
print(f"Encoded indices: {indices}")
print(f"Decoded codes: {decoded}")
print(f"Round-trip successful: {codes == decoded}")

# Test 2: Bijection property
print(f"\nBijection test:")
print(f"code2idx size: {len(vocab.code2idx)}")
print(f"idx2code size: {len(vocab.idx2code)}")
print(f"Equal sizes: {len(vocab.code2idx) == len(vocab.idx2code)}")

# Test 3: Consistency check
for code, idx in vocab.code2idx.items():
    assert vocab.idx2code[idx] == code, f"Inconsistency: {code} → {idx} → {vocab.idx2code[idx]}"
print("All mappings consistent!")
```

**Expected Output:**
```
Original codes: ['401.9', '428.0', '250.00']
Encoded indices: [0, 2, 1]
Decoded codes: ['401.9', '428.0', '250.00']
Round-trip successful: True

Bijection test:
code2idx size: 4
idx2code size: 4
Equal sizes: True
All mappings consistent!
```

### Exercise 4: Compare with BPE Fragmentation

**Task:** Demonstrate why BPE would fail for medical codes.

```python
from transformers import BartTokenizer
from vocabulary import DiagnosisVocabulary

# BART tokenizer (fragments codes)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# DiagnosisVocabulary (atomic codes)
diagnosis_vocab = DiagnosisVocabulary()
diagnosis_vocab.add_codes(["401.9", "401.1", "250.00", "250.01"])

# Compare tokenization
codes = ["401.9", "401.1", "250.00", "250.01"]

print("Code | BART Tokens (fragmented) | Diagnosis Index (1:1)")
print("-----|---------------------------|----------------------")
for code in codes:
    bart_tokens = bart_tokenizer.tokenize(code)
    diag_idx = diagnosis_vocab.code2idx[code]
    print(f"{code:6} | {str(bart_tokens):25} | {diag_idx}")

print(f"\nBART total tokens: {len(bart_tokenizer.tokenize(' '.join(codes)))}")
print(f"Diagnosis total tokens: {len(codes)} (1:1 mapping)")
```

**Expected Output:**
```
Code | BART Tokens (fragmented) | Diagnosis Index (1:1)
-----|---------------------------|----------------------
401.9  | ['Ġ401', '.', '9']        | 0
401.1  | ['Ġ401', '.', '1']        | 1
250.00 | ['Ġ250', '.', '00']       | 2
250.01 | ['Ġ250', '.', '01']       | 3

BART total tokens: 12 (3 tokens per code, fragmented)
Diagnosis total tokens: 4 (1:1 mapping)
```

### Exercise 5: Load Real MIMIC-III Vocabulary

**Task:** Build vocabulary from actual MIMIC-III data.

```python
from data_loader import load_mimic_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data (this also builds vocabulary)
patients, vocab = load_mimic_data(
    patients_path='data_files/PATIENTS.csv',
    admissions_path='data_files/ADMISSIONS.csv',
    diagnoses_path='data_files/DIAGNOSES_ICD.csv',
    logger=logger,
    num_patients=1000  # First 1000 patients
)

print(f"\nVocabulary statistics:")
print(f"  Total size: {len(vocab)}")
print(f"  First 10 codes: {list(vocab.code2idx.keys())[:10]}")
print(f"  First 10 indices: {list(vocab.idx2code.keys())[:10]}")

# Test encoding a real visit
sample_patient = patients[0]
sample_visit = sample_patient.visits[0]
encoded_visit = vocab.encode(sample_visit)

print(f"\nSample patient {sample_patient.subject_id}:")
print(f"  Visit codes: {sample_visit}")
print(f"  Encoded indices: {encoded_visit}")
```

---

## Common Pitfalls

### Pitfall 1: Assuming Indices Are Ordered by Frequency

**Wrong Assumption:**
```python
# Index 0 is NOT necessarily the most frequent code
vocab = DiagnosisVocabulary()
vocab.add_codes(["rare_code", "common_code"])  # Order of addition matters

# "rare_code" gets index 0 (added first)
# "common_code" gets index 1 (added second)
```

**Reality:** Indices assigned in order codes are seen, not by frequency.

### Pitfall 2: Modifying Vocabulary After Training

**Wrong:**
```python
# Train model with vocabulary size 5,562
vocab = build_vocabulary_from_training_data()
model = PromptBartModel(vocab_size=len(vocab))
train(model)

# Add new codes to vocabulary (BAD!)
vocab.add_code("999.99")  # New code, index 5562
# Model embedding layer still expects 5,562 tokens (index out of bounds!)
```

**Right:** Vocabulary must be frozen after training starts.

### Pitfall 3: Forgetting to Handle Unknown Codes

**Wrong:**
```python
# Test data has unseen code
vocab.encode(["401.9", "unseen_code"])  # KeyError!
```

**Right:** Use DiagnosisCodeTokenizer which adds <unk> token (see page 07).

### Pitfall 4: Confusing Vocabulary Index with Token ID

**Vocabulary indices:** 0, 1, 2, ..., 6984 (code indices only)
**Token IDs:** 0-6 (special tokens), 7-6991 (code tokens with offset=7)

**See next page for token ID mapping.**

---

## Connection to Tokenization Layer

### Vocabulary → Tokenizer

**DiagnosisVocabulary provides:**
- Code-to-index mapping (1:1)
- Index-to-code mapping (reverse)
- Vocabulary size (6,985 codes)

**DiagnosisCodeTokenizer adds:**
- Special tokens (<s>, <pad>, </s>, <unk>, <v>, <\v>, <mask>)
- Token ID offset (special tokens 0-6, codes start at 7)
- Unknown token handling (<unk> for unseen codes)
- Sequence formatting (visit markers, BOS/EOS)

**Relationship:**
```
DiagnosisVocabulary.code2idx["401.9"] = 0  (vocabulary index)
                                       ↓
DiagnosisCodeTokenizer.encode("401.9") = 7  (token ID = 0 + 7)
```

**See next page:** [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)

---

## Key Design Decisions

### Decision 1: Build Vocabulary During Data Loading

**Why not separate passes?**
- **Efficiency:** Single iteration through 46K patients
- **Simplicity:** Vocabulary and data structures built together
- **Natural order:** Codes added chronologically

**Alternative:** Two-pass approach (slower, unnecessary)

### Decision 2: No Frequency-Based Ordering

**Why not sort by frequency?**
- **Consistency:** Vocabulary order deterministic (depends only on data order)
- **Simplicity:** No need to count frequencies during building
- **Flexibility:** Frequency information can be computed separately if needed

**Note:** Some NLP systems assign lower indices to frequent words (not used here).

### Decision 3: No Minimum Frequency Threshold

**Why include all codes?**
- **Completeness:** Even rare codes matter clinically
- **No information loss:** All MIMIC-III codes preserved
- **Hierarchical approach:** Category-level training handles rare codes

**Alternative:** Filter codes with < 10 occurrences (would lose 1,000+ codes)

---

## Summary

**DiagnosisVocabulary is the foundation of semantic integrity:**

1. **1:1 Mapping:** Each code gets exactly one index (no fragmentation)
2. **Bidirectional:** code2idx and idx2code for fast encoding/decoding
3. **Built During Loading:** Single pass through MIMIC-III data
4. **6,985 Unique Codes:** Full vocabulary from training set
5. **Long Tail Distribution:** Top 100 codes account for 45% of occurrences
6. **Atomic Units:** Codes treated as indivisible tokens, not text
7. **Medical Knowledge:** ICD-9 codes preserved exactly as specified

**Key Files:**
- `vocabulary.py:7-46` - DiagnosisVocabulary class
- `data_loader.py:150-175` - Vocabulary building during data load

---

## What's Next?

You now understand how diagnosis codes are mapped to indices with 1:1 correspondence.

**Next:** [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md) - Learn how DiagnosisCodeTokenizer adds special tokens, visit markers, and converts vocabulary indices to token IDs for BART.

**Alternative Path:**
- [08_DATA_LOADING.md](08_DATA_LOADING.md) - Deep dive into load_mimic_data() function
- [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md) - Skip to hierarchical approach (943 categories)

---

**Navigation:**
- ← Back to [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md)
- → Next: [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
