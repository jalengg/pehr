# Data Pipeline

**Last Updated:** October 24, 2025

This document describes how MIMIC-III data flows from raw CSVs to tokenized batches ready for model training.

## Overview

```
MIMIC-III CSVs → PatientRecords → Vocabulary → Tokenization → Dataset → Corruption → Batches
```

## 1. MIMIC-III Data Sources

**Location:** `data_files/` (gzip compressed CSV files)

### Files Used

| File | Purpose | Key Columns |
|------|---------|-------------|
| `PATIENTS.csv.gz` | Patient demographics | `SUBJECT_ID`, `GENDER`, `DOB` |
| `ADMISSIONS.csv.gz` | Hospital admissions | `SUBJECT_ID`, `HADM_ID`, `ADMITTIME`, `ETHNICITY` |
| `DIAGNOSES_ICD.csv.gz` | ICD-9 diagnosis codes | `SUBJECT_ID`, `HADM_ID`, `ICD9_CODE`, `SEQ_NUM` |

### Files Available But Unused

- `PROCEDURES_ICD.csv.gz` - Procedure codes (future work)
- `PRESCRIPTIONS.csv.gz` - Medications (future work)
- `ICUSTAYS.csv.gz` - ICU stays (future work)

## 2. Data Loading (data_loader.py)

### Function: `load_mimic_data()`

**Input:**
- Paths to MIMIC-III CSV files
- `num_patients`: Maximum patients to load (default: 25,000)

**Process:**

1. **Load CSVs**
   ```python
   patients_df = pd.read_csv('PATIENTS.csv.gz')
   admissions_df = pd.read_csv('ADMISSIONS.csv.gz')
   diagnoses_df = pd.read_csv('DIAGNOSES_ICD.csv.gz')
   ```

2. **Merge DataFrames**
   ```python
   # Join patients → admissions → diagnoses
   merged = patients_df.merge(admissions_df, on='SUBJECT_ID')
   merged = merged.merge(diagnoses_df, on=['SUBJECT_ID', 'HADM_ID'])
   ```

3. **Calculate Age**
   ```python
   # Age at first admission
   age = (admit_time - dob).days / 365.25
   age = min(age, 90)  # Privacy cap (MIMIC-III protocol)
   ```

4. **Group by Patient and Visit**
   ```python
   # Each patient can have multiple admissions
   # Each admission can have multiple diagnosis codes
   visits = [[code1, code2, ...], [code3, code4, ...], ...]
   ```

5. **Sort Visits Chronologically**
   ```python
   visits.sort(key=lambda v: v.admit_time)
   ```

6. **Extract Demographics**
   ```python
   demographics = {
       'age': float,        # Continuous: 0-90
       'gender': int,       # Categorical: 0=M, 1=F
       'ethnicity': int     # Categorical: 0-5 (removed in current version)
   }
   ```

**Output:**
- `patient_records`: List of `PatientRecord` objects
- `vocab`: `DiagnosisVocabulary` instance

### PatientRecord Structure

```python
@dataclass
class PatientRecord:
    subject_id: str
    visits: List[List[str]]  # List of visits, each visit is list of ICD-9 codes
    age: float               # Age at first admission (0-90)
    gender: int              # 0=M, 1=F
    ethnicity: int           # Deprecated (was 0-5 for different ethnicities)
```

**Example:**
```python
PatientRecord(
    subject_id="12345",
    visits=[
        ["401.9", "250.00", "585.9"],  # Visit 1: Hypertension, Diabetes, CKD
        ["428.0", "584.9"]              # Visit 2: Heart failure, Acute kidney failure
    ],
    age=65.3,
    gender=0,  # Male
    ethnicity=0  # Deprecated
)
```

## 3. Vocabulary Creation (vocabulary.py)

### DiagnosisVocabulary Class

**Purpose:** 1:1 mapping between ICD-9 code strings and integer IDs

**Implementation:**
```python
class DiagnosisVocabulary:
    def __init__(self):
        self.code2idx: Dict[str, int] = {}
        self.idx2code: Dict[int, str] = {}
        self._next_idx = 0

    def add_code(self, code: str) -> int:
        if code not in self.code2idx:
            idx = self._next_idx
            self.code2idx[code] = idx
            self.idx2code[idx] = code
            self._next_idx += 1
        return self.code2idx[code]
```

**Example Mapping:**
```
"V3001"  → 0
"250.00" → 1
"401.9"  → 2
"428.0"  → 3
...
(5562 unique codes total)
```

**Why This Design?**
- Each code = single integer ID (no fragmentation)
- Prevents BART tokenizer from splitting "401.9" into ["401", ".", "9"]
- Enables embedding lookup: code ID → learnable vector

## 4. Tokenization (code_tokenizer.py)

### DiagnosisCodeTokenizer Class

**Purpose:** Combine BART special tokens with diagnosis code vocabulary

**Token ID Assignment:**

```python
# Special tokens (0-6)
0: <s>       # Start of sequence (BART BOS)
1: <pad>     # Padding token
2: </s>      # End of sequence (BART EOS)
3: <unk>     # Unknown token
4: <v>       # Start of visit marker
5: <\v>      # End of visit marker
6: <mask>    # Mask token (for infilling)

# Diagnosis codes (7+)
7:  "V3001"  # Code 0 from vocabulary
8:  "250.00" # Code 1 from vocabulary
9:  "401.9"  # Code 2 from vocabulary
...
5568: [last code]
```

**Code offset:** 7 (special tokens occupy 0-6)

**Key Methods:**

```python
def encode_visits(visits: List[List[str]]) -> List[int]:
    """
    Convert visits to token IDs.

    Input: [["401.9", "250.00"], ["428.0"]]
    Output: [4, 9, 8, 5, 4, 10, 5, 2]
             <v> 401.9 250.00 <\v> <v> 428.0 <\v> </s>
    """

def decode(token_ids: List[int]) -> str:
    """
    Convert token IDs back to human-readable string.

    Input: [4, 9, 8, 5, 2]
    Output: "<v> 401.9 250.00 <\v> </s>"
    """
```

**Token Sequence Format:**
```
<s> <v> code1 code2 ... <\v> <v> code3 code4 ... <\v> </s>
│   │                    │   │                    │   │
│   └─ Visit 1 start     │   └─ Visit 2 start     │   └─ End of sequence
└─ Start of sequence     └─ Visit 1 end           └─ Visit 2 end
```

## 5. Dataset (dataset.py)

### EHRPatientDataset

**Purpose:** PyTorch dataset wrapper for patient records

```python
class EHRPatientDataset(Dataset):
    def __init__(self, patient_records: List[PatientRecord]):
        self.patient_records = patient_records

    def __getitem__(self, idx: int) -> PatientRecord:
        return self.patient_records[idx]

    def __len__(self) -> int:
        return len(self.patient_records)
```

**Simple wrapper:** Just stores and retrieves patient records. All processing happens in the collator.

## 6. Data Corruption (dataset.py)

### CorruptionFunctions Class

**Purpose:** Generate diverse training samples via data augmentation

**Three Corruption Types:**

#### 6.1 Mask Infilling

**Purpose:** BART-style span masking for denoising objective

**Process:**
1. Sample span length from Poisson distribution (λ=3.0)
2. Replace span of codes with single `<mask>` token
3. Model learns to reconstruct masked codes

**Example:**
```python
Original:  <v> 401.9 250.00 585.9 428.0 <\v>
Masked:    <v> 401.9 <mask> <\v>
Label:     <v> 401.9 250.00 585.9 428.0 <\v>
```

#### 6.2 Token Deletion

**Purpose:** Robustness to missing codes

**Process:**
1. Randomly delete codes with probability p=0.15
2. Ensure at least 1 code remains per visit

**Example:**
```python
Original:  <v> 401.9 250.00 585.9 <\v>
Deleted:   <v> 401.9 585.9 <\v>  # 250.00 deleted
Label:     <v> 401.9 250.00 585.9 <\v>
```

#### 6.3 Token Replacement

**Purpose:** Learn to distinguish valid from invalid codes

**Process:**
1. Replace codes with random alternatives from vocabulary with p=0.15

**Example:**
```python
Original:  <v> 401.9 250.00 <\v>
Replaced:  <v> 401.9 V3001 <\v>  # 250.00 → random code V3001
Label:     <v> 401.9 250.00 <\v>
```

#### 6.4 Next-Visit Prediction

**Purpose:** Temporal perplexity evaluation

**Process:**
1. For patients with 2+ visits, randomly select split point N
2. Input: visits[0:N-1] + `<mask>`
3. Label: visits[0:N] (includes next visit)

**Example:**
```python
Original: <v> 401.9 <\v> <v> 428.0 <\v> <v> 584.9 <\v>
Input:    <v> 401.9 <\v> <mask>
Label:    <v> 401.9 <\v> <v> 428.0 <\v>
```

## 7. Data Collation (dataset.py)

### EHRDataCollator

**Purpose:** Batch patient records, apply corruptions, pad sequences

**Process:**

1. **Sample Expansion:** Each patient → 1-5 training samples
   - Always: Teacher forcing (original, uncorrupted)
   - Probabilistic (50% each): Mask infilling, deletion, replacement
   - Conditional (if 2+ visits): Next-visit prediction

2. **Tokenization:** Convert visits to token IDs via `DiagnosisCodeTokenizer`

3. **Padding:** Pad all sequences to max length in batch

4. **Label Creation:**
   - Copy input_ids
   - Set padding tokens to -100 (ignored by loss)
   - For mask infilling, only compute loss on masked positions

5. **Batching:** Stack into tensors

**Output:**
```python
{
    'x_num': torch.FloatTensor,      # [batch, 1] ages
    'x_cat': torch.LongTensor,       # [batch, 1] genders
    'input_ids': torch.LongTensor,   # [batch, max_len] token IDs
    'attention_mask': torch.LongTensor,  # [batch, max_len] mask
    'labels': torch.LongTensor       # [batch, max_len] targets (-100 for padding)
}
```

## Configuration (config.py)

### Relevant Parameters

```python
@dataclass
class DataConfig:
    patients_path: str = "data_files/PATIENTS.csv.gz"
    admissions_path: str = "data_files/ADMISSIONS.csv.gz"
    diagnoses_path: str = "data_files/DIAGNOSES_ICD.csv.gz"
    num_patients: int = 25000
    train_val_split: float = 0.2  # 20% validation

@dataclass
class TrainingConfig:
    # Corruption probabilities
    lambda_poisson: float = 3.0     # Span length for masking
    mask_probability: float = 0.15  # Mask infilling prob
    del_probability: float = 0.15   # Token deletion prob
    rep_probability: float = 0.15   # Token replacement prob
    corruption_prob: float = 0.5    # Apply each corruption 50% of time

    # Corruption flags
    use_mask_infilling: bool = True
    use_token_deletion: bool = True
    use_token_replacement: bool = True
    use_next_visit_prediction: bool = True
```

## Data Statistics

**MIMIC-III (25,000 patients):**
- Vocabulary size: 5,562 unique ICD-9 codes
- Average visits per patient: 1.30
- Average codes per visit: 9.15
- Gender distribution: 56% male, 44% female
- Age distribution: Mean ~60 years, capped at 90

**Token Sequence Statistics:**
- Average sequence length: ~25 tokens (including special tokens)
- Max sequence length: 512 (configurable, truncated if longer)
- Effective batch size: 3-4x nominal (due to sample expansion)

## Common Issues

### Issue 1: Code Fragmentation

**Problem:** BART tokenizer fragments ICD-9 codes
```
"401.9" → ["401", ".", "9"]  # WRONG
```

**Solution:** DiagnosisCodeTokenizer assigns each code a single token ID
```
"401.9" → [9]  # Correct (single token)
```

### Issue 2: Empty Visits After Corruption

**Problem:** Deletion/masking can remove all codes from a visit

**Solution:** CorruptionFunctions ensures at least 1 code or `<mask>` remains per visit

### Issue 3: Padding in Labels

**Problem:** Model learns to predict padding tokens

**Solution:** Set padding positions to -100 in labels (ignored by CrossEntropyLoss)

## Next Steps

- **Understand model architecture:** See [Model Architecture](03_MODEL_ARCHITECTURE.md)
- **Learn about training:** See [Training](04_TRAINING.md)
- **See how data flows through model:** See [Architecture Overview](01_ARCHITECTURE.md)
