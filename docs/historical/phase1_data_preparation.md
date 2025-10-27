# Phase 1: Data Preparation - Implementation Summary

**Date:** October 12, 2025
**Status:** ✅ Complete
**Objective:** Separate demographics from medical codes, create code-specific vocabulary with no fragmentation

---

## Overview

Phase 1 fundamentally restructures data representation to match PromptEHR's architecture:

**Before (main.py text-based approach):**
```python
sequence = "65 WHITE M <demo> <v> 401.9 250.00 <\v> <END>"
# Everything mixed in one text string
# BART tokenizer fragments codes: 401.9 → ['401', '.', '9']
```

**After (Phase 1 structured approach):**
```python
patient = {
    'x_num': [65.0],              # Age as continuous feature
    'x_cat': [0, 0],              # Gender=M(0), Ethnicity=WHITE(0)
    'visits': [['401.9', '250.00']]  # Codes as vocabulary indices
}
# Demographics separated from codes
# Each code = single token ID (no fragmentation)
```

---

## Files Created

### 1. `vocabulary.py` - DiagnosisVocabulary Class

**Purpose:** 1:1 mapping between diagnosis codes and integer IDs

**Key Methods:**
```python
class DiagnosisVocabulary:
    def add_code(self, code: str) -> int:
        # Assigns unique integer ID to each code
        # "V3001" → 0, "250.00" → 1, etc.

    def encode(self, codes: List[str]) -> List[int]:
        # Convert code strings to IDs
        # ["V3001", "250.00"] → [0, 1]

    def decode(self, indices: List[int]) -> List[str]:
        # Convert IDs back to codes
        # [0, 1] → ["V3001", "250.00"]
```

**Why This Works:**
- Direct string-to-integer mapping (no subword tokenization)
- Bidirectional dictionaries: `code2idx` and `idx2code`
- Each code is **exactly one token** (critical difference from BART tokenizer)

**Example:**
```python
vocab = DiagnosisVocabulary()
vocab.add_code("V3001")  # Returns 0
vocab.add_code("250.00") # Returns 1
vocab.add_code("401.9")  # Returns 2

# No fragmentation:
codes = ["V3001", "250.00", "401.9"]
encoded = vocab.encode(codes)  # [0, 1, 2]
# 3 codes → 3 IDs (vs BART's 8-10 subword tokens)
```

---

### 2. `data_loader.py` - MIMIC-III Data Loading

**Purpose:** Load MIMIC-III data and separate demographics from visit sequences

**Key Classes:**

#### PatientRecord
```python
class PatientRecord:
    def __init__(
        self,
        subject_id: int,
        age: float,              # Continuous numerical feature
        gender: str,             # Categorical: M/F
        ethnicity: str,          # Categorical: WHITE/BLACK/etc.
        visits: List[List[str]]  # [[visit1_codes], [visit2_codes], ...]
    )

    def to_dict(self) -> Dict:
        return {
            'x_num': np.array([age]),           # [1] continuous features
            'x_cat': np.array([gender_id, ethnicity_id]),  # [2] categorical IDs
            'visits': visits,                   # List of code lists
            'num_visits': len(visits)
        }
```

**Categorical Encoding:**
```python
# Gender: M=0, F=1
# Ethnicity: Mapped to indices in ETHNICITY_CATEGORIES
ETHNICITY_CATEGORIES = [
    'WHITE',                    # 0
    'BLACK',                    # 1
    'HISPANIC OR LATINO',       # 2
    'ASIAN',                    # 3
    'OTHER',                    # 4
    'UNKNOWN/NOT SPECIFIED'     # 5
]

# Example patient:
age = 65.0
gender = "M"  → 0
ethnicity = "WHITE"  → 0

x_num = [65.0]
x_cat = [0, 0]
```

#### load_mimic_data Function
```python
def load_mimic_data(
    patients_path, admissions_path, diagnoses_path, logger, num_patients
) -> Tuple[List[PatientRecord], DiagnosisVocabulary]:
    # 1. Load MIMIC-III CSVs
    patients_df = pd.read_csv(patients_path)
    admissions_df = pd.read_csv(admissions_path)
    diagnoses_df = pd.read_csv(diagnoses_path)

    # 2. Calculate age at first admission
    age = (first_admission_date - date_of_birth).years
    age = min(age, 90)  # Privacy: cap at 90

    # 3. Normalize ethnicity to canonical categories
    ethnicity = normalize_ethnicity(raw_ethnicity_string)

    # 4. Group diagnoses by visit (HADM_ID)
    visits = [[code1, code2], [code3], ...]

    # 5. Build vocabulary as we load data
    vocab.add_codes(all_codes_for_visit)

    # 6. Create PatientRecord for each patient
    records.append(PatientRecord(subject_id, age, gender, ethnicity, visits))

    return records, vocab
```

**Data Flow:**
```
MIMIC-III CSVs
    ↓
Merge: PATIENTS + ADMISSIONS + DIAGNOSES_ICD
    ↓
Calculate age, normalize ethnicity
    ↓
Group by SUBJECT_ID → Group by HADM_ID
    ↓
PatientRecord(age=65, gender=M, visits=[[V3001], [250.00, 401.9]])
    ↓
DiagnosisVocabulary (built during loading)
```

---

### 3. `code_tokenizer.py` - DiagnosisCodeTokenizer

**Purpose:** Convert diagnosis codes and structural tokens to integer IDs

**Special Tokens:**
```python
PAD_TOKEN = "<PAD>"     # ID: 0 (padding)
BOS_TOKEN = "<BOS>"     # ID: 1 (beginning of sequence)
EOS_TOKEN = "<EOS>"     # ID: 2 (end of sequence - BART compatibility)
V_TOKEN = "<v>"         # ID: 3 (visit start)
V_END_TOKEN = "<\\v>"   # ID: 4 (visit end)
END_TOKEN = "<END>"     # ID: 5 (patient sequence end)

# Medical codes start at ID 6+
code_offset = 6
```

**Key Methods:**

#### encode_codes
```python
def encode_codes(self, codes: List[str]) -> List[int]:
    # Encode medical codes to token IDs
    vocab_ids = self.vocab.encode(codes)  # Get vocab indices
    return [idx + self.code_offset for idx in vocab_ids]

# Example:
codes = ["V3001", "250.00"]
vocab_ids = [0, 1]  # From vocabulary
token_ids = [6, 7]  # Add offset of 6
```

#### encode_visit
```python
def encode_visit(self, codes: List[str], add_markers=True) -> List[int]:
    # Encode single visit with structural markers
    code_ids = self.encode_codes(codes)
    if add_markers:
        return [V_TOKEN_ID] + code_ids + [V_END_TOKEN_ID]

# Example:
codes = ["V3001", "250.00"]
token_ids = [3, 6, 7, 4]  # <v> V3001 250.00 <\v>
```

#### encode_patient
```python
def encode_patient(self, visits: List[List[str]], add_special_tokens=True):
    # Encode full patient sequence
    token_ids = [BOS_TOKEN_ID]
    for visit in visits:
        visit_ids = self.encode_visit(visit, add_markers=True)
        token_ids.extend(visit_ids)
    token_ids.append(END_TOKEN_ID)
    return token_ids

# Example:
visits = [["V3001", "250.00"], ["401.9"]]
token_ids = [
    1,           # <BOS>
    3, 6, 7, 4,  # <v> V3001 250.00 <\v>
    3, 8, 4,     # <v> 401.9 <\v>
    5            # <END>
]
# Total: 9 tokens for 3 diagnosis codes + structure
```

**Token ID Ranges:**
```
0-5:   Special tokens (PAD, BOS, EOS, <v>, <\v>, <END>)
6+:    Medical codes (offset by 6 from vocabulary indices)
```

---

### 4. `dataset.py` - PyTorch Dataset and DataCollator

**Purpose:** PyTorch-compatible dataset and batching with padding

#### EHRPatientDataset
```python
class EHRPatientDataset(Dataset):
    def __getitem__(self, idx) -> Dict:
        record = self.patient_records[idx]

        # Encode visits to token sequence
        token_ids = self.tokenizer.encode_patient(record.visits)

        return {
            'x_num': record.age as [1] array,
            'x_cat': [gender_id, ethnicity_id] as [2] array,
            'visit_codes': record.visits (raw codes),
            'token_ids': encoded token sequence,
            'subject_id': record.subject_id
        }
```

**Sample Output:**
```python
dataset[0] = {
    'x_num': array([65.]),
    'x_cat': array([0, 0]),  # M, WHITE
    'visit_codes': [['V3001', 'V053'], ['250.00', '401.9']],
    'token_ids': array([1, 3, 6, 7, 4, 3, 8, 9, 4, 5]),
    'subject_id': 12345
}
```

#### EHRDataCollator
```python
class EHRDataCollator:
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Stack demographics (same shape)
        x_num = torch.stack([item['x_num'] for item in batch])
        x_cat = torch.stack([item['x_cat'] for item in batch])

        # 2. Pad token sequences to max_seq_length
        for item in batch:
            tokens = item['token_ids']
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]  # Truncate
            else:
                # Pad with PAD_TOKEN_ID (0)
                padding = [0] * (max_seq_length - len(tokens))
                tokens = tokens + padding

            # 3. Create attention mask (1 for real, 0 for padding)
            attention_mask = [1] * len(tokens) + [0] * num_padding

            # 4. Create labels (mask padding with -100)
            labels = tokens.copy()
            labels[labels == PAD_TOKEN_ID] = -100

        return {
            'x_num': x_num,              # [batch, 1]
            'x_cat': x_cat,              # [batch, 2]
            'input_ids': input_ids,      # [batch, max_seq_len]
            'attention_mask': mask,      # [batch, max_seq_len]
            'labels': labels             # [batch, max_seq_len]
        }
```

**Collation Example:**
```python
# Input batch (2 patients):
batch = [
    {'token_ids': [1, 3, 6, 7, 4, 5], 'x_num': [65.], 'x_cat': [0, 0]},
    {'token_ids': [1, 3, 8, 9, 4, 3, 10, 4, 5], 'x_num': [42.], 'x_cat': [1, 2]}
]

# Output (padded to max_seq_length=256):
{
    'x_num': [[65.], [42.]],  # [2, 1]
    'x_cat': [[0, 0], [1, 2]],  # [2, 2]
    'input_ids': [
        [1, 3, 6, 7, 4, 5, 0, 0, 0, ...],           # 6 real + 250 padding
        [1, 3, 8, 9, 4, 3, 10, 4, 5, 0, 0, ...]     # 9 real + 247 padding
    ],  # [2, 256]
    'attention_mask': [
        [1, 1, 1, 1, 1, 1, 0, 0, 0, ...],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
    ],  # [2, 256]
    'labels': [
        [1, 3, 6, 7, 4, 5, -100, -100, ...],
        [1, 3, 8, 9, 4, 3, 10, 4, 5, -100, ...]
    ]  # [2, 256] - padding masked with -100
}
```

**Why -100 for labels?**
PyTorch CrossEntropyLoss ignores targets with value -100, so loss is only computed on real tokens, not padding.

---

### 5. `test_phase1.py` - Validation Tests

**Purpose:** Verify no fragmentation and correct data structures

**Tests:**

1. **Vocabulary Test**
   - Add codes: `["V3001", "250.00", "401.9"]`
   - Verify 3 codes → 3 IDs (no fragmentation)
   - Test encode/decode roundtrip

2. **Tokenizer Test**
   - Encode visit: `["V3001", "250.00"]` → `[3, 6, 7, 4]`
   - Encode patient: 2 visits → 9 tokens
   - Decode preserves intact codes

3. **PatientRecord Test**
   - Verify x_num shape: `[1]`
   - Verify x_cat shape: `[2]`
   - Check gender/ethnicity encoding

4. **No Fragmentation Test (Critical)**
   - Test challenging codes: `["V3001", "250.00", "401.9", "E950.4", "99.04", "V053"]`
   - Verify 6 codes → 6 token IDs (1:1 mapping)
   - Roundtrip decode produces exact original codes

**All tests pass ✅**

---

## Autoregressive Generation Loop (Future Phase 4)

**How generation will work with this data structure:**

```python
def generate_patient(model, x_num, x_cat, num_visits):
    # Step 1: Encode demographics as prompt embeddings (NOT tokens)
    prompt_embeds = prompt_encoder(x_num, x_cat)
    # Shape: [batch=1, n_features=3, hidden_dim=768]
    # [age_embedding, gender_embedding, ethnicity_embedding]

    # Step 2: Initialize sequence
    input_ids = [BOS_TOKEN_ID]  # [1]

    # Step 3: Generate visits autoregressively
    for visit_idx in range(num_visits):
        # Generate visit start token
        input_ids.append(V_TOKEN_ID)  # [1, 3]

        # Generate diagnosis codes for this visit
        while True:
            # Forward pass with prompt conditioning
            outputs = model(
                input_ids=torch.tensor([input_ids]),
                prompt_embeds=prompt_embeds,  # Demographics as embeddings
                attention_mask=torch.ones(len(input_ids))
            )

            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :]

            # Sample next token (temperature, top-k, etc.)
            next_token_id = sample(next_token_logits, temp=1.0, top_k=50)

            # If visit end token, break
            if next_token_id == V_END_TOKEN_ID:
                input_ids.append(V_END_TOKEN_ID)
                break

            # Otherwise, add diagnosis code token
            input_ids.append(next_token_id)

        # Visit complete: input_ids = [1, 3, 6, 7, 4]
        #                              BOS <v> code1 code2 <\v>

    # Step 4: Add final END token
    input_ids.append(END_TOKEN_ID)

    # Step 5: Decode to diagnosis codes
    codes_generated = tokenizer.decode(input_ids)
    return codes_generated
```

**Key Differences from Old Approach:**

| Aspect | Old (main.py) | New (Phase 1) |
|--------|---------------|---------------|
| Demographics | Text tokens in sequence | Continuous embeddings (x_num, x_cat) |
| Medical codes | BART subwords | Vocabulary indices (1:1 mapping) |
| Fragmentation | 3 codes → 8-10 tokens | 3 codes → 3 tokens |
| Conditioning | Demographics in input_ids | Demographics as prompt_embeds |
| Structure | Learned from text | Enforced by generation loop |

**Autoregressive Context Example:**

```python
# Visit 1 generation:
Input:  [BOS] + prompt_embeds([65.0], [0, 0])
Output: [3, 6, 7, 4]  # <v> V3001 250.00 <\v>

# Visit 2 generation (sees previous visit):
Input:  [BOS, 3, 6, 7, 4] + prompt_embeds([65.0], [0, 0])
Output: [3, 8, 4]  # <v> 401.9 <\v>
#       ↑ Model knows: after diabetes (250.00), hypertension (401.9) is likely

# Visit 3 generation (sees full history):
Input:  [BOS, 3, 6, 7, 4, 3, 8, 4] + prompt_embeds([65.0], [0, 0])
Output: [3, 9, 10, 4]  # <v> 428.0 585.6 <\v>
#       ↑ Model learns progression: diabetes + hypertension → heart + kidney failure
```

**Temporal Order Preserved:**
- Visits are generated sequentially
- Each visit sees all previous visits via attention
- Model learns disease progression patterns implicitly
- No explicit time gaps (could be added to x_num if needed)

---

## Key Achievements

### 1. No Fragmentation ✅
```python
# Before (BART tokenizer):
"V3001" → ['V', '300', '1']  # 3 tokens
"250.00" → ['250', '.', '00']  # 3 tokens
# 2 codes → 6 tokens (fragmented)

# After (DiagnosisVocabulary):
"V3001" → [6]   # 1 token
"250.00" → [7]  # 1 token
# 2 codes → 2 tokens (1:1 mapping)
```

### 2. Demographics Separated ✅
```python
# Before:
sequence = "65 WHITE M <demo> <v> 401.9 <\v> <END>"
# Demographics mixed with codes

# After:
x_num = [65.0]
x_cat = [0, 0]  # M, WHITE
visits = [['401.9']]
# Demographics separate, ready for prompt embedding
```

### 3. Vocabulary Size Controlled ✅
```python
# BART vocabulary: 50,265 pretrained tokens
# Our vocabulary: 6 special tokens + ~8,000 diagnosis codes = ~8,006 tokens
# Much smaller, focused on medical domain
```

### 4. Data Structures Ready for Phase 2 ✅
- PatientRecord: structured patient data
- DiagnosisCodeTokenizer: maps codes to IDs
- EHRDataCollator: batches with proper padding/masking
- All ready for PromptBartModel in Phase 2

---

## Codebase Knowledge Map

```
pehr_scratch/
│
├── main.py.copy                 # Backup of original implementation
│
├── Phase 1 (Data Preparation) - COMPLETE ✅
│   ├── vocabulary.py            # DiagnosisVocabulary: code ↔ ID mapping
│   ├── data_loader.py           # MIMIC-III loading, PatientRecord class
│   ├── code_tokenizer.py        # DiagnosisCodeTokenizer: codes → token IDs
│   ├── dataset.py               # EHRPatientDataset, EHRDataCollator
│   └── test_phase1.py           # Validation tests (all pass)
│
├── Phase 2 (Model Architecture) - IN PROGRESS 🔨
│   ├── conditional_prompt.py    # NumericalPrompt, CategoricalPrompt (TODO)
│   ├── prompt_bart_encoder.py   # Custom BART encoder (TODO)
│   ├── prompt_bart_decoder.py   # Custom BART decoder (TODO)
│   └── prompt_bart_model.py     # Full PromptBartModel (TODO)
│
├── Phase 3 (Training Pipeline) - PENDING ⏳
│   ├── trainer.py               # Training loop with prompts (TODO)
│   └── metrics.py               # Perplexity, code accuracy (TODO)
│
├── Phase 4 (Generation) - PENDING ⏳
│   ├── generator.py             # Autoregressive generation (TODO)
│   └── sampling.py              # Temperature, top-k, constraints (TODO)
│
├── Phase 5 (Evaluation) - PENDING ⏳
│   └── evaluate.py              # Distribution analysis (TODO)
│
├── Documentation
│   ├── promptehr_comparison.md  # PromptEHR vs our approach
│   ├── phase1_summary.md        # This file
│   └── CLAUDE.md                # Project overview
│
└── Data Files
    ├── PATIENTS.csv.gz
    ├── ADMISSIONS.csv.gz
    └── DIAGNOSES_ICD.csv.gz
```

---

## Class Hierarchy

```
Data Layer:
  DiagnosisVocabulary
      ↓
  DiagnosisCodeTokenizer
      ↓
  PatientRecord → EHRPatientDataset → EHRDataCollator
                                          ↓
                                      DataLoader

Model Layer (Phase 2):
  ConditionalPrompt (x_num, x_cat → embeddings)
      ↓
  PromptBartEncoder (adds prompt embeddings)
      ↓
  PromptBartDecoder (adds prompt embeddings)
      ↓
  PromptBartModel (full seq2seq with prompts)

Training Layer (Phase 3):
  PromptBartModel + DataLoader → Trainer → Checkpoints

Generation Layer (Phase 4):
  PromptBartModel + Demographics → Generator → Synthetic Patients
```

---

## Next Steps: Phase 2 - Model Architecture

**Objectives:**
1. Implement `ConditionalPrompt` module
   - `NumericalConditionalPrompt(x_num)` → age embedding
   - `CategoricalConditionalPrompt(x_cat)` → gender/ethnicity embeddings
   - Combined prompt: `[age_emb, gender_emb, ethnicity_emb]`

2. Modify BART encoder to accept prompt embeddings
   - Prepend prompt embeddings to input embeddings
   - Adjust attention mask for prepended prompts

3. Modify BART decoder to accept prompt embeddings
   - Same prepending mechanism as encoder

4. Create full PromptBartModel
   - Combines PromptBartEncoder + PromptBartDecoder
   - Forward pass with x_num, x_cat → prompt_embeds

5. Test forward pass with dummy data
   - Verify shapes, no errors
   - Check loss computation

**Estimated time:** 2 days

---

**Phase 1 Status:** ✅ COMPLETE
**All tests passing:** ✅
**Ready for Phase 2:** ✅

---

**Prepared by:** Claude Code
**Date:** October 12, 2025
