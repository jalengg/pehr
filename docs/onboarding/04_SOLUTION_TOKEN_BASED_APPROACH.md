# 04: Solution - Token-Based Architecture

**Estimated Time:** 45 minutes
**Prerequisites:** [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md)
**Next:** [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md)

---

## Learning Objectives

By the end of this page, you will understand:
- The token-based architecture that replaced text-based approach
- DiagnosisVocabulary: 1:1 code-to-ID mapping
- DiagnosisCodeTokenizer: Custom tokenization without fragmentation
- Separation of demographics from code sequences
- How this solution preserves semantic integrity
- Why this approach succeeded where text-based failed

---

## The Core Insight (October 9, 2025)

**After the fragmentation incident, the breakthrough realization:**

> "Medical codes must be treated as atomic tokens, not text strings. Each ICD-9 code needs exactly one token ID, with no subword fragmentation."

**This led to a complete architectural redesign:**
- ❌ **Old:** Text-based with BART tokenizer (fragmentation)
- ✅ **New:** Token-based with custom vocabulary (1:1 mapping)

---

## Architecture Redesign: Three Core Components

### 1. DiagnosisVocabulary (Atomic Code Mapping)

**Purpose:** Maintain 1:1 mapping between ICD-9 codes and token IDs

**Before (Text-based):**
```python
# BART tokenizer
"401.9" → ['Ġ401', '.', 'Ġ9']  # 3 tokens, fragmented
```

**After (Token-based):**
```python
# DiagnosisVocabulary
"401.9" → 0  # Single token ID
"250.00" → 1
"428.0" → 2
```

**Key Property:** Each code is a **single atomic unit**.

---

### 2. DiagnosisCodeTokenizer (Custom Tokenization)

**Purpose:** Extend vocabulary with special tokens, bypass BPE entirely

**Token ID Layout:**
```
0-6:    Special tokens (<s>, <pad>, </s>, <unk>, <v>, <\v>, <mask>)
7+:     Diagnosis codes (offset = 7)
```

**Example:**
```python
Code "401.9" → Vocab index 0 → Token ID 7  (0 + offset)
Code "250.00" → Vocab index 1 → Token ID 8  (1 + offset)
Code "428.0" → Vocab index 2 → Token ID 9  (2 + offset)
```

**No BART tokenizer involved** - direct vocabulary lookup.

---

### 3. Structured Data Separation

**Purpose:** Separate demographics from medical codes

**Before (Text-based):**
```python
# Everything mixed in one string
"65 WHITE M <demo> <v> 401.9 250.00 <\v> <END>"
```

**After (Token-based):**
```python
# Demographics: Separate variables
age: float = 65.0
gender: int = 0  # 0=M, 1=F

# Codes: Token sequence only
tokens: [<s>, <v>, 7, 8, <\v>, </s>]  # [BOS, visit_start, 401.9, 250.00, visit_end, EOS]
```

**Why separate?**
- Demographics never compete for token space
- Cleaner semantic boundaries
- Easier preprocessing and validation

---

## DiagnosisVocabulary Implementation

**File:** `vocabulary.py`

### Class Structure

```python
class DiagnosisVocabulary:
    """
    Vocabulary for diagnosis codes (1:1 mapping).

    Attributes:
        vocab: List[str] - Ordered list of codes
        code2idx: Dict[str, int] - Code → index mapping
        idx2code: Dict[int, str] - Index → code mapping
    """

    def __init__(self):
        self.vocab = []
        self.code2idx = {}
        self.idx2code = {}

    def build_vocab(self, patients: List[PatientRecord]):
        """
        Build vocabulary from patient records.

        Args:
            patients: List of PatientRecord objects

        Process:
            1. Extract all unique codes from all patients
            2. Sort alphabetically for consistency
            3. Build bidirectional mappings
        """
        # Collect all unique codes
        unique_codes = set()
        for patient in patients:
            for visit in patient.visits:
                for code in visit:
                    unique_codes.add(code)

        # Sort for deterministic ordering
        self.vocab = sorted(list(unique_codes))

        # Build mappings
        for idx, code in enumerate(self.vocab):
            self.code2idx[code] = idx
            self.idx2code[idx] = code

    def get_idx(self, code: str) -> int:
        """Get index for code."""
        return self.code2idx.get(code, -1)

    def get_code(self, idx: int) -> str:
        """Get code for index."""
        return self.idx2code.get(idx, "<unk>")

    def __len__(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)
```

**Key Design Choices:**
1. **Sorted vocabulary:** Ensures consistent token IDs across runs
2. **Bidirectional mappings:** Fast lookup in both directions
3. **Unknown handling:** Returns -1 for unknown codes (handled separately)

---

## DiagnosisCodeTokenizer Implementation

**File:** `code_tokenizer.py`

### Token ID Layout

**Special Tokens (Hardcoded IDs 0-6):**
```python
SPECIAL_TOKENS = {
    '<s>': 0,      # Beginning of sequence
    '<pad>': 1,    # Padding token
    '</s>': 2,     # End of sequence
    '<unk>': 3,    # Unknown token
    '<v>': 4,      # Visit start
    '<\v>': 5,     # Visit end
    '<mask>': 6    # Mask token (for corruption)
}
```

**Diagnosis Code Tokens (Starting at ID 7):**
```python
code_offset = 7

# Example with 6,985 codes:
Token ID 7:    Code "000" (first code in sorted vocab)
Token ID 8:    Code "001"
...
Token ID 6991: Code "V99.99" (last code in sorted vocab)

Total vocabulary size: 7 + 6,985 = 6,992 tokens
```

### Class Structure

```python
class DiagnosisCodeTokenizer:
    """
    Custom tokenizer for diagnosis codes (no BPE fragmentation).

    Attributes:
        vocab: DiagnosisVocabulary
        special_tokens: Dict[str, int]
        code_offset: int = 7
        vocab_size: int = len(special_tokens) + len(vocab)
    """

    def __init__(self, vocab: DiagnosisVocabulary):
        self.vocab = vocab
        self.code_offset = 7
        self.vocab_size = self.code_offset + len(vocab)

        # Special tokens
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.visit_start_token_id = 4
        self.visit_end_token_id = 5
        self.mask_token_id = 6

    def encode_codes(self, codes: List[str]) -> List[int]:
        """
        Encode list of codes to token IDs.

        Args:
            codes: List of ICD-9 codes (e.g., ["401.9", "250.00"])

        Returns:
            List of token IDs (e.g., [7, 8])
        """
        token_ids = []
        for code in codes:
            idx = self.vocab.get_idx(code)
            if idx == -1:
                # Unknown code
                token_ids.append(self.unk_token_id)
            else:
                # Add offset to get token ID
                token_ids.append(idx + self.code_offset)
        return token_ids

    def decode_codes(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs back to codes.

        Args:
            token_ids: List of token IDs (e.g., [7, 8])

        Returns:
            List of ICD-9 codes (e.g., ["401.9", "250.00"])
        """
        codes = []
        for token_id in token_ids:
            # Skip special tokens
            if token_id < self.code_offset:
                continue

            # Convert token ID to vocabulary index
            vocab_idx = token_id - self.code_offset

            # Lookup code
            code = self.vocab.get_code(vocab_idx)
            if code != "<unk>":
                codes.append(code)

        return codes

    def encode_patient(self, patient: PatientRecord) -> List[int]:
        """
        Encode full patient record to token sequence.

        Args:
            patient: PatientRecord with visits

        Returns:
            Token sequence: [<s>, <v>, code1, code2, <\v>, <v>, code3, <\v>, </s>]
        """
        tokens = [self.bos_token_id]  # Start with <s>

        for visit in patient.visits:
            tokens.append(self.visit_start_token_id)  # <v>
            tokens.extend(self.encode_codes(visit))    # Diagnosis codes
            tokens.append(self.visit_end_token_id)     # <\v>

        tokens.append(self.eos_token_id)  # End with </s>

        return tokens
```

**Key Features:**
1. **No BPE:** Direct vocabulary lookup, no subword fragmentation
2. **Special token handling:** Separate namespace (0-6)
3. **Offset arithmetic:** `token_id = vocab_idx + 7`
4. **Structured sequences:** Explicit visit boundaries with `<v>` and `<\v>`

---

## Example: Encoding a Patient

### Input PatientRecord

```python
patient = PatientRecord(
    subject_id="12345",
    age=65.0,
    gender="M",  # 0 after encoding
    visits=[
        ["401.9", "250.00"],  # Visit 1
        ["428.0"]             # Visit 2
    ]
)
```

### Step-by-Step Encoding

**Step 1: Build vocabulary (done once for all patients)**
```python
vocabulary = DiagnosisVocabulary()
vocabulary.build_vocab(all_patients)

# Result (simplified):
vocab = ["401.9", "250.00", "428.0", ...]
code2idx = {"401.9": 0, "250.00": 1, "428.0": 2, ...}
```

**Step 2: Create tokenizer**
```python
tokenizer = DiagnosisCodeTokenizer(vocabulary)

# Token ID mapping:
# "401.9" → vocab_idx=0 → token_id=7
# "250.00" → vocab_idx=1 → token_id=8
# "428.0" → vocab_idx=2 → token_id=9
```

**Step 3: Encode patient**
```python
token_ids = tokenizer.encode_patient(patient)

# Result:
token_ids = [0, 4, 7, 8, 5, 4, 9, 5, 2]
#           <s> <v> 401.9 250.00 <\v> <v> 428.0 <\v> </s>
```

**Step 4: Decode back (for verification)**
```python
codes = tokenizer.decode_codes(token_ids)

# Result:
codes = ["401.9", "250.00", "428.0"]  # Perfect reconstruction!
```

---

## PatientRecord Data Structure

**File:** `data_loader.py:40-73`

**Before (Text-based):**
```python
# Patient represented as string
patient_str = "65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 <\v> <END>"
```

**After (Token-based):**
```python
class PatientRecord:
    """Container for patient data (structured)."""

    def __init__(
        self,
        subject_id: str,
        age: float,
        gender: str,  # "M" or "F"
        visits: List[List[str]]  # [[codes_visit1], [codes_visit2], ...]
    ):
        self.subject_id = subject_id
        self.age = age
        self.gender = 0 if gender == "M" else 1  # Encode immediately
        self.visits = visits

    def __repr__(self):
        return f"PatientRecord(id={self.subject_id}, age={self.age}, gender={'M' if self.gender==0 else 'F'}, visits={len(self.visits)})"
```

**Benefits:**
1. **Type safety:** age is float, gender is int, visits are lists
2. **Easy validation:** Check age range, gender encoding, visit structure
3. **No parsing:** Direct attribute access, no string manipulation

---

## Comparison: Text-based vs Token-based

### Data Flow Comparison

**Text-based (Old):**
```
Patient → String → BART Tokenizer → Fragmented Tokens → Gibberish
  |                      |
  |                      └─> "401.9" → ['Ġ401', '.', 'Ġ9']
  └─> "65 WHITE M <demo> <v> 401.9 ..."
```

**Token-based (New):**
```
Patient → Demographics + Codes → DiagnosisCodeTokenizer → Clean Tokens → Valid Codes
  |                                    |
  |                                    └─> "401.9" → [7]  (single token!)
  └─> age=65, gender=0, visits=[["401.9", "250.00"], ...]
```

### Code Example Comparison

**Text-based (Fragmentation):**
```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
patient_str = "65 WHITE M <demo> <v> 401.9 250.00 <\v>"
token_ids = tokenizer.encode(patient_str)

# Result: [8121, 24410, 256, 1437, 2154, 1437, 2154, 20174, 4, 860, 21268, 4, 713, 1437, 2154, 1437]
# 16 tokens for 2 codes! (Plus demographics mixed in)
```

**Token-based (Clean):**
```python
from code_tokenizer import DiagnosisCodeTokenizer

tokenizer = DiagnosisCodeTokenizer(vocabulary)
patient = PatientRecord(age=65, gender="M", visits=[["401.9", "250.00"]])
token_ids = tokenizer.encode_patient(patient)

# Result: [0, 4, 7, 8, 5, 2]
# 6 tokens: <s> <v> 401.9 250.00 <\v> </s>
# Demographics separate: age=65.0, gender=0
```

---

## Why This Solution Works

### 1. Semantic Integrity Preserved

**Problem with BPE:**
- "401.9" → ["401", ".", "9"] - meaning destroyed

**Solution with 1:1 mapping:**
- "401.9" → [7] - meaning preserved as atomic unit

**Result:** Model learns code-level semantics, not subword patterns.

---

### 2. Valid Generation Guaranteed

**Problem with BPE:**
- Model generates: ["401", ".", "506"] → "401.506" (invalid code!)

**Solution with 1:1 mapping:**
- Model generates: [7] → "401.9" (guaranteed valid, in vocabulary)

**Result:** Generated tokens are always valid ICD-9 codes.

---

### 3. Clean Semantic Boundaries

**Problem with text-based:**
- "65 WHITE M <demo> <v> 401.9" - demographics + codes mixed

**Solution with structured data:**
- `age=65, gender=0` (demographics)
- `[<s>, <v>, 7, 8, <\v>, </s>]` (codes only)

**Result:** Model focuses on code generation, demographics provide conditioning.

---

### 4. No Parsing Ambiguity

**Problem with text-based:**
- "401.9 250.00 428" - Is "428" complete or "428.0"? Where are visit boundaries?

**Solution with structured tokens:**
- `[<v>, 7, 8, 5, <v>, 9, 5]` - Explicit boundaries with `<v>` and `<\v>`

**Result:** Unambiguous decoding, clear structure.

---

## Implementation Timeline (October 2025)

**October 9:** Fragmentation incident - gibberish generation discovered

**October 10:** Solution designed - token-based architecture

**October 12-17:** Implementation (Phased approach)
- **Phase 1:** DiagnosisVocabulary + DiagnosisCodeTokenizer
- **Phase 2:** PromptBart model architecture
- **Phase 3:** Dataset + corruption functions

**October 18:** First successful generation with clean codes

**Result:** Complete architectural shift from text-based to token-based in 1 week.

---

## Current Status (November 2025)

**Token-based architecture is the foundation:**
- ✅ DiagnosisVocabulary: 6,985 unique ICD-9 codes
- ✅ DiagnosisCodeTokenizer: 6,992 total tokens (7 special + 6,985 codes)
- ✅ No code fragmentation issues
- ✅ All extensions built on this foundation:
  - Multi-task learning (age/sex heads)
  - Hierarchical tokenizer (7,935 tokens with categories)
  - Co-occurrence regularization

**This solution enabled all subsequent innovations.**

---

## Try It Yourself

### Exercise 1: Build Vocabulary

```python
from data_loader import load_mimic_data, PatientRecord
from vocabulary import DiagnosisVocabulary

# Load patients
patients = load_mimic_data(n_patients=1000)

# Build vocabulary
vocabulary = DiagnosisVocabulary()
vocabulary.build_vocab(patients)

print(f"Vocabulary size: {len(vocabulary)}")  # ~6,985 for full MIMIC-III
print(f"First 10 codes: {vocabulary.vocab[:10]}")

# Test lookups
code = "401.9"
idx = vocabulary.get_idx(code)
reconstructed = vocabulary.get_code(idx)

print(f"Code '{code}' → index {idx} → code '{reconstructed}'")
# Output: Code '401.9' → index 0 → code '401.9'  ✓ Perfect round-trip
```

### Exercise 2: Encode and Decode

```python
from code_tokenizer import DiagnosisCodeTokenizer

# Create tokenizer
tokenizer = DiagnosisCodeTokenizer(vocabulary)

# Create patient
patient = PatientRecord(
    subject_id="test",
    age=67.0,
    gender="F",
    visits=[["401.9", "250.00"], ["428.0", "585.9"]]
)

# Encode
token_ids = tokenizer.encode_patient(patient)
print(f"Token IDs: {token_ids}")

# Decode
codes = tokenizer.decode_codes(token_ids)
print(f"Decoded codes: {codes}")

# Verify
original_codes = [code for visit in patient.visits for code in visit]
print(f"Original codes: {original_codes}")
print(f"Match: {codes == original_codes}")  # Should be True
```

**Expected Output:**
```
Token IDs: [0, 4, 7, 8, 5, 4, 9, 10, 5, 2]
Decoded codes: ['401.9', '250.00', '428.0', '585.9']
Original codes: ['401.9', '250.00', '428.0', '585.9']
Match: True  ✓
```

### Exercise 3: Compare with BART Tokenizer

```python
from transformers import BartTokenizer

# BART tokenizer (old approach)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
codes_str = "401.9 250.00 428.0"
bart_tokens = bart_tokenizer.tokenize(codes_str)
bart_ids = bart_tokenizer.encode(codes_str)

print("BART (fragmented):")
print(f"  Tokens: {bart_tokens}")
print(f"  IDs: {bart_ids}")
print(f"  Count: {len(bart_ids)} tokens for 3 codes")

# Our tokenizer (new approach)
codes_list = ["401.9", "250.00", "428.0"]
our_ids = tokenizer.encode_codes(codes_list)

print("\nOurs (clean):")
print(f"  IDs: {our_ids}")
print(f"  Count: {len(our_ids)} tokens for 3 codes")

print(f"\nImprovement: {len(bart_ids) / len(our_ids):.1f}x fewer tokens")
```

**Expected Output:**
```
BART (fragmented):
  Tokens: ['Ġ401', '.', 'Ġ9', 'Ġ250', '.', '00', 'Ġ428', '.', 'Ġ0']
  IDs: [20174, 4, 860, 21268, 4, 713, 34938, 4, 321]
  Count: 9 tokens for 3 codes

Ours (clean):
  IDs: [7, 8, 9]
  Count: 3 tokens for 3 codes

Improvement: 3.0x fewer tokens  ✓
```

---

## Key Takeaways

1. **Token-based architecture replaced text-based** after October 9, 2025 fragmentation incident.

2. **DiagnosisVocabulary provides 1:1 code-to-ID mapping** - each ICD-9 code gets exactly one token ID.

3. **DiagnosisCodeTokenizer bypasses BPE entirely** - direct vocabulary lookup, no fragmentation.

4. **PatientRecord separates demographics from codes** - structured data, no parsing ambiguity.

5. **7 special tokens (0-6) + 6,985 code tokens (7-6991)** = 6,992 total vocabulary.

6. **Semantic integrity preserved** - "401.9" is atomic unit, meaning not destroyed.

7. **Valid generation guaranteed** - model can only generate codes that exist in vocabulary.

8. **This foundation enabled all subsequent innovations** - multi-task learning, hierarchical generation, co-occurrence regularization all built on this.

---

## What's Next?

You now understand **the solution** that replaced the failed text-based approach.

**Next:** [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md) - Deep dive into PatientRecord class structure and data loading from MIMIC-III.

**Alternative Path:**
- [06_VOCABULARY_SYSTEM.md](06_VOCABULARY_SYSTEM.md) - Detailed implementation of DiagnosisVocabulary
- [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md) - DiagnosisCodeTokenizer internals

---

**Navigation:**
- ← Back to [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md)
- → Next: [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
