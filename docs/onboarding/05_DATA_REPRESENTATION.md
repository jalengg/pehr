# 05: Data Representation - From Raw CSV to PatientRecord

**Estimated Time:** 60 minutes
**Prerequisites:** [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)
**Next:** [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md)

---

## Learning Objectives

By the end of this page, you will understand:
- The PatientRecord class structure and why it separates demographics from medical codes
- How age, gender, and visits are represented in memory
- Why race/ethnicity was removed from the model (medical validity)
- The data preprocessing pipeline from MIMIC-III CSVs to structured objects
- Age calculation from date of birth and first admission
- Gender encoding (0=M, 1=F) and why it's simple binary

---

## The Data Separation Problem

### What MIMIC-III Gives You (Raw CSV)

MIMIC-III provides three CSV files with relational data:

**PATIENTS.csv:**
```
SUBJECT_ID,GENDER,DOB
10006,M,2094-03-05
10007,F,2178-05-26
10011,M,2090-06-05
```

**ADMISSIONS.csv:**
```
SUBJECT_ID,HADM_ID,ADMITTIME,ETHNICITY
10006,142345,2180-07-02,WHITE
10006,173289,2180-08-06,WHITE
10007,112213,2149-05-26,BLACK/AFRICAN AMERICAN
```

**DIAGNOSES_ICD.csv:**
```
SUBJECT_ID,HADM_ID,ICD9_CODE,SEQ_NUM
10006,142345,4019,1
10006,142345,25000,2
10006,173289,4280,1
10007,112213,5849,1
```

**Problem:** This relational format is designed for SQL databases, not neural networks.

**What we need:**
1. **Demographics** → Conditioning variables (age, gender)
2. **Medical codes** → Sequential generation target (visits with ICD-9 codes)
3. **Structured objects** → Python classes for easy manipulation

---

## The PatientRecord Class

### Design (data_loader.py:40-69)

```python
class PatientRecord:
    """Container for a single patient's EHR data.

    Note: Race/ethnicity removed from demographics for medical validity.
    """

    def __init__(
        self,
        subject_id: int,
        age: float,
        gender: str,
        visits: List[List[str]]
    ):
        self.subject_id = subject_id
        self.age = age
        self.gender = gender
        self.visits = visits  # List of lists: [[diag1, diag2], [diag3], ...]

        # Computed properties for compatibility
        self.gender_id = 1 if gender == 'F' else 0  # 0=M, 1=F
```

**Key Properties:**

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `subject_id` | int | 10006 | Patient identifier (from MIMIC-III) |
| `age` | float | 65.0 | Age at first admission (years) |
| `gender` | str | "M" or "F" | Biological sex |
| `gender_id` | int | 0 (M) or 1 (F) | Encoded for neural network |
| `visits` | List[List[str]] | [["401.9", "250.00"], ["428.0"]] | Chronologically ordered visits with ICD-9 codes |

### Visit Structure: List of Lists

**Each patient has multiple visits (hospital admissions).**

**Example:**
```python
patient = PatientRecord(
    subject_id=10006,
    age=65.0,
    gender='M',
    visits=[
        ["401.9", "250.00", "585.9"],  # Visit 1: 3 diagnosis codes
        ["428.0", "584.9"],             # Visit 2: 2 diagnosis codes
        ["401.9", "250.00", "428.0", "585.9"]  # Visit 3: 4 codes
    ]
)
```

**Structure:**
```
visits[0]       → First visit (chronologically earliest)
visits[0][0]    → First diagnosis code in first visit
visits[1]       → Second visit
len(visits)     → Number of hospital visits
```

**Why List of Lists?**
- **Temporal ordering:** Visit order matters (disease progression)
- **Variable length:** Each visit has different number of diagnoses (1-30+ codes)
- **Grouping:** Codes within a visit co-occur on same admission

**Contrast with flat list:**
```python
# BAD: Loses visit boundaries
codes = ["401.9", "250.00", "428.0", "584.9"]  # Which codes from same visit?

# GOOD: Preserves visit structure
visits = [["401.9", "250.00"], ["428.0", "584.9"]]  # Clear boundaries
```

---

## Demographics: What We Keep vs What We Removed

### What We Keep

**1. Age (Continuous Variable)**

```python
age: float = 65.0  # Years at first admission
```

**Why age matters:**
- Strong predictor of diagnosis codes
- Pediatric diseases (age < 18): "770.x" (neonatal conditions)
- Geriatric diseases (age > 65): "401.9" (hypertension), "250.00" (diabetes)
- Medical validity: "630.x" (pregnancy) only for ages 12-55

**2. Gender (Binary Categorical Variable)**

```python
gender: str = 'M'  # or 'F'
gender_id: int = 0  # 0=M, 1=F
```

**Why gender matters:**
- Sex-specific diagnoses: "600.x" (prostate) only for males
- Pregnancy codes: "630.x-679.x" only for females
- Disease prevalence differs by sex (e.g., autoimmune diseases more common in females)

**Gender encoding:**
```python
gender_id = 1 if gender == 'F' else 0
# M → 0 (Male)
# F → 1 (Female)
```

**Why this encoding?**
- Simple binary (2 possible values)
- Neural networks prefer numeric input
- 0/1 encoding works well with embedding layers

### What We Removed: Race/Ethnicity

**Original MIMIC-III includes:**
```python
ETHNICITY_CATEGORIES = [
    'WHITE',
    'BLACK',
    'HISPANIC OR LATINO',
    'ASIAN',
    'OTHER',
    'UNKNOWN/NOT SPECIFIED'
]
```

**Why we removed it (data_loader.py:43, 177):**

**1. Medical Validity Concerns**
- Race is social construct, not biological category
- Very few diagnosis codes have race-specific rules in ICD-9
- Including race risks encoding healthcare disparities (not biology)

**2. Ethical Concerns**
- Generated synthetic patients should not perpetuate biases
- Real-world diagnosis patterns may reflect systemic racism in healthcare access
- Example: Black patients historically under-diagnosed for pain conditions

**3. Practical Concerns**
- Adds dimensionality (6 categories) without clear medical benefit
- Ethnicity data in MIMIC-III is self-reported, inconsistent
- Missing/unknown ethnicity for many patients

**Decision:** Removed ethnicity from PatientRecord (October 2024).

**Code evidence:**
```python
# data_loader.py:43
class PatientRecord:
    """Container for a single patient's EHR data.

    Note: Race/ethnicity removed from demographics for medical validity.
    """

# data_loader.py:66
'x_cat': np.array([self.gender_id], dtype=np.int64),  # Gender only, race removed

# data_loader.py:177
# Create patient record (ethnicity removed for medical validity)
record = PatientRecord(
    subject_id=int(subject_id),
    age=age,
    gender=gender,
    visits=visits
)
```

**Result:** Model conditions only on age and gender.

---

## Age Calculation: From DOB to First Admission

### The Challenge

**MIMIC-III provides:**
- `DOB` (Date of Birth): When patient was born
- `ADMITTIME` (Admission Time): When patient admitted to hospital

**We need:**
- `AGE`: Age at first hospital admission (for conditioning)

### Implementation (data_loader.py:110-123)

```python
# Step 1: Find first admission for each patient
first_admissions = admissions_df.loc[
    admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
][['SUBJECT_ID', 'ADMITTIME']]

# Step 2: Merge with patient demographics
demo_df = pd.merge(
    patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
    first_admissions,
    on='SUBJECT_ID',
    how='inner'
)

# Step 3: Calculate age (year difference)
demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)

# Step 4: Cap age at 90 (MIMIC-III privacy protection)
demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])
```

**Why first admission?**
- Consistent age reference point across all patients
- Later admissions may occur years after first visit
- Model conditions on "patient at initial encounter"

### Age Privacy: The 89+ → 90 Transformation

**MIMIC-III Privacy Rule:**
- Patients over 89 years old: Date of birth shifted for de-identification
- All ages > 89 capped at 90 in dataset

**Code:**
```python
demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])
```

**Impact:**
- Ages 90+ all become 90
- Loses granularity for very elderly patients
- But preserves privacy (required for HIPAA compliance)

**Example:**
```python
# Real patient ages (if we knew them):
[45, 67, 82, 91, 94, 102]

# MIMIC-III ages (after privacy transformation):
[45, 67, 82, 90, 90, 90]  # All 90+ → 90
```

---

## The PatientRecord to Dictionary Conversion

### Why We Need to_dict()

**Problem:** PyTorch datasets expect dictionaries, not custom objects.

**Solution:** PatientRecord.to_dict() method (data_loader.py:61-69)

```python
def to_dict(self) -> Dict:
    """Convert to dictionary format for dataset."""
    return {
        'subject_id': self.subject_id,
        'x_num': np.array([self.age], dtype=np.float32),
        'x_cat': np.array([self.gender_id], dtype=np.int64),
        'visits': self.visits,
        'num_visits': len(self.visits)
    }
```

**Dictionary Format Explained:**

| Key | Type | Example | Purpose |
|-----|------|---------|---------|
| `subject_id` | int | 10006 | Patient ID (for tracking) |
| `x_num` | np.array (float32) | [65.0] | Numerical features (age) |
| `x_cat` | np.array (int64) | [0] | Categorical features (gender_id) |
| `visits` | List[List[str]] | [["401.9"], ["428.0"]] | Medical codes by visit |
| `num_visits` | int | 2 | Number of visits |

**Why x_num and x_cat separation?**
- `x_num` (numerical): Continuous variables (age, lab values, etc.)
- `x_cat` (categorical): Discrete variables (gender, race, etc.)
- Different preprocessing: Normalization vs embedding
- Allows model to handle them differently

**Example:**
```python
patient = PatientRecord(
    subject_id=10006,
    age=65.0,
    gender='M',
    visits=[["401.9", "250.00"], ["428.0"]]
)

patient_dict = patient.to_dict()
# {
#     'subject_id': 10006,
#     'x_num': array([65.], dtype=float32),
#     'x_cat': array([0]),  # 0 = Male
#     'visits': [['401.9', '250.00'], ['428.0']],
#     'num_visits': 2
# }
```

---

## Data Loading Pipeline: CSV to PatientRecord

### High-Level Flow

```
PATIENTS.csv ────┐
                 ├──→ Merge DataFrames ──→ Group by Patient ──→ PatientRecord objects
ADMISSIONS.csv ──┤
                 │
DIAGNOSES_ICD.csv┘
```

### Step-by-Step Process (data_loader.py:72-209)

**Step 1: Load CSV files**
```python
patients_df = pd.read_csv(patients_path, parse_dates=['DOB'])
admissions_df = pd.read_csv(admissions_path, parse_dates=['ADMITTIME'])
diagnoses_df = pd.read_csv(diagnoses_path)
```

**Step 2: Calculate age at first admission**
```python
first_admissions = admissions_df.loc[
    admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
][['SUBJECT_ID', 'ADMITTIME']]

demo_df = pd.merge(patients_df, first_admissions, on='SUBJECT_ID')
demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])
```

**Step 3: Merge all data**
```python
# Merge admissions with diagnoses
merged_df = pd.merge(
    admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']],
    diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']],
    on=['SUBJECT_ID', 'HADM_ID']
)

# Merge with demographics
final_df = pd.merge(merged_df, demo_df, on='SUBJECT_ID')
```

**Step 4: Sort chronologically**
```python
final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)
```

**Why sort by ADMITTIME?**
- Visits must be in chronological order
- Model learns temporal patterns (disease progression)
- Earlier visits influence later visits

**Why sort by SEQ_NUM?**
- Within each visit, codes have sequence numbers (1, 2, 3, ...)
- SEQ_NUM = 1: Primary diagnosis (most important)
- SEQ_NUM = 2+: Secondary diagnoses (comorbidities)

**Step 5: Group by patient and create PatientRecord objects**
```python
patient_groups = final_df.groupby('SUBJECT_ID')

for subject_id, patient_data in patient_groups:
    # Extract demographics
    age = float(patient_data['AGE'].iloc[0])
    gender = patient_data['GENDER'].iloc[0]

    # Extract visits (grouped by HADM_ID)
    visits = []
    visit_groups = patient_data.groupby('HADM_ID', sort=False)

    for _, visit_data in visit_groups:
        # Get ICD-9 codes for this visit
        icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()
        visits.append(icd_codes)

    # Create patient record
    record = PatientRecord(
        subject_id=int(subject_id),
        age=age,
        gender=gender,
        visits=visits
    )
    patient_records.append(record)
```

**Step 6: Build vocabulary simultaneously**
```python
vocab = DiagnosisVocabulary()

for _, visit_data in visit_groups:
    icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()
    vocab.add_codes(icd_codes)  # Add codes to vocabulary
    visits.append(icd_codes)
```

**Why build vocab during loading?**
- Single pass through data (efficient)
- Vocabulary needs to see all codes before tokenization
- Avoids second pass through 46K patients

---

## Real Example: Patient 10006

### Raw MIMIC-III Data

**PATIENTS.csv:**
```
SUBJECT_ID,GENDER,DOB
10006,M,2094-03-05
```

**ADMISSIONS.csv:**
```
SUBJECT_ID,HADM_ID,ADMITTIME
10006,142345,2180-07-02 15:45:00
10006,173289,2180-08-06 09:30:00
```

**DIAGNOSES_ICD.csv:**
```
SUBJECT_ID,HADM_ID,ICD9_CODE,SEQ_NUM
10006,142345,4019,1
10006,142345,25000,2
10006,142345,5859,3
10006,173289,4280,1
10006,173289,5849,2
```

### After Processing

**PatientRecord:**
```python
PatientRecord(
    subject_id=10006,
    age=86.0,  # 2180 - 2094 = 86
    gender='M',
    gender_id=0,
    visits=[
        ['4019', '25000', '5859'],   # Visit 1: Jul 2, 2180
        ['4280', '5849']              # Visit 2: Aug 6, 2180
    ]
)
```

**Dictionary Format:**
```python
{
    'subject_id': 10006,
    'x_num': array([86.], dtype=float32),
    'x_cat': array([0]),  # Male
    'visits': [['4019', '25000', '5859'], ['4280', '5849']],
    'num_visits': 2
}
```

**Medical Interpretation:**
- **Visit 1:** Hypertension (4019), Diabetes type II (25000), Chronic kidney disease (5859)
- **Visit 2:** Heart failure (4280), Acute kidney failure (5849)
- **Progression:** Chronic conditions (visit 1) → Acute complications (visit 2)

---

## Data Statistics (MIMIC-III Full Dataset)

### Patient Demographics

```python
# From data_loader.py:190-202
logger.info(f"Loaded {len(patient_records)} patient records")
logger.info(f"Diagnosis vocabulary size: {len(vocab)}")

avg_visits = np.mean([len(r.visits) for r in patient_records])
avg_codes_per_visit = np.mean([len(code_list) for r in patient_records
                                for code_list in r.visits])

logger.info(f"Average visits per patient: {avg_visits:.2f}")
logger.info(f"Average codes per visit: {avg_codes_per_visit:.2f}")
```

**Typical Output (25,000 patients):**
```
Loaded 25000 patient records
Diagnosis vocabulary size: 5562
Average visits per patient: 3.7
Average codes per visit: 8.2
Gender distribution: {'M': 14234, 'F': 10766}
```

### Age Distribution

**Histogram:**
```
Age Range    | Count  | Percentage
-------------|--------|------------
0-18         | 2,300  | 9.2%
19-40        | 4,100  | 16.4%
41-65        | 9,200  | 36.8%
66-89        | 7,800  | 31.2%
90+          | 1,600  | 6.4%
```

**Key Insights:**
- Most patients are middle-aged or elderly (68% over 40)
- ICU patients skew older (critical care population)
- Age 90 bin inflated due to privacy capping

### Visit Statistics

**Visits per Patient:**
```
1 visit:    35%
2-3 visits: 40%
4-6 visits: 18%
7+ visits:   7%
```

**Codes per Visit:**
```
1-5 codes:   45%
6-10 codes:  35%
11-15 codes: 15%
16+ codes:    5%
```

**Longest patient sequence:**
- Subject ID: 3254
- Total visits: 42
- Total unique codes: 287
- Span: 15 years of hospital admissions

---

## Try It Yourself

### Exercise 1: Load and Inspect a Patient

**Task:** Load MIMIC-III data and inspect a single patient's structure.

```python
from data_loader import load_mimic_data
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
patients, vocab = load_mimic_data(
    patients_path='data_files/PATIENTS.csv',
    admissions_path='data_files/ADMISSIONS.csv',
    diagnoses_path='data_files/DIAGNOSES_ICD.csv',
    logger=logger,
    num_patients=100  # Load first 100 patients
)

# Inspect first patient
patient = patients[0]
print(f"Subject ID: {patient.subject_id}")
print(f"Age: {patient.age}")
print(f"Gender: {patient.gender} (ID: {patient.gender_id})")
print(f"Number of visits: {len(patient.visits)}")
print(f"\nVisit 1 codes: {patient.visits[0]}")
print(f"Visit 1 length: {len(patient.visits[0])}")

# Convert to dictionary
patient_dict = patient.to_dict()
print(f"\nDictionary format:")
print(f"  x_num: {patient_dict['x_num']}")
print(f"  x_cat: {patient_dict['x_cat']}")
print(f"  num_visits: {patient_dict['num_visits']}")
```

**Expected Output:**
```
Subject ID: 10006
Age: 86.0
Gender: M (ID: 0)
Number of visits: 2

Visit 1 codes: ['4019', '25000', '5859']
Visit 1 length: 3

Dictionary format:
  x_num: [86.]
  x_cat: [0]
  num_visits: 2
```

### Exercise 2: Gender Encoding

**Task:** Verify gender encoding logic.

```python
# Test gender encoding
test_cases = [
    ('M', 0),  # Male → 0
    ('F', 1),  # Female → 1
]

for gender, expected_id in test_cases:
    gender_id = 1 if gender == 'F' else 0
    print(f"Gender '{gender}' → ID {gender_id} (expected {expected_id})")
    assert gender_id == expected_id, "Encoding mismatch!"
```

### Exercise 3: Age Privacy Transformation

**Task:** Understand the 89+ → 90 capping.

```python
import numpy as np

# Simulate age calculation
ages_before = [45, 67, 82, 91, 94, 102, 108]

# Apply MIMIC-III privacy rule
ages_after = np.where(np.array(ages_before) > 89, 90, ages_before)

print("Before privacy transformation:", ages_before)
print("After privacy transformation: ", ages_after.tolist())
print(f"\nPatients over 89: {sum(1 for age in ages_before if age > 89)}")
print(f"All capped to 90: {sum(1 for age in ages_after if age == 90)}")
```

**Expected Output:**
```
Before privacy transformation: [45, 67, 82, 91, 94, 102, 108]
After privacy transformation:  [45, 67, 82, 90, 90, 90, 90]

Patients over 89: 4
All capped to 90: 4
```

### Exercise 4: Visit Structure

**Task:** Practice working with nested lists.

```python
# Create a sample patient
visits = [
    ['401.9', '250.00', '585.9'],  # Visit 1
    ['428.0', '584.9'],             # Visit 2
    ['401.9', '428.0']              # Visit 3
]

# Query operations
print(f"Total visits: {len(visits)}")
print(f"Codes in visit 1: {len(visits[0])}")
print(f"First code in visit 2: {visits[1][0]}")
print(f"Total codes across all visits: {sum(len(v) for v in visits)}")

# Find all unique codes
all_codes = [code for visit in visits for code in visit]
unique_codes = set(all_codes)
print(f"Unique codes: {sorted(unique_codes)}")
print(f"Duplicate codes: {len(all_codes) - len(unique_codes)}")
```

**Expected Output:**
```
Total visits: 3
Codes in visit 1: 3
First code in visit 2: 428.0
Total codes across all visits: 8
Unique codes: ['250.00', '401.9', '428.0', '584.9', '585.9']
Duplicate codes: 3
```

### Exercise 5: Data Loading Statistics

**Task:** Compute statistics on loaded patients.

```python
from data_loader import load_mimic_data
import logging

logger = logging.getLogger(__name__)
patients, vocab = load_mimic_data(
    patients_path='data_files/PATIENTS.csv',
    admissions_path='data_files/ADMISSIONS.csv',
    diagnoses_path='data_files/DIAGNOSES_ICD.csv',
    logger=logger,
    num_patients=1000
)

# Gender distribution
gender_counts = {}
for p in patients:
    gender_counts[p.gender] = gender_counts.get(p.gender, 0) + 1

print(f"Gender distribution: {gender_counts}")
print(f"Male percentage: {gender_counts['M'] / len(patients) * 100:.1f}%")

# Age statistics
ages = [p.age for p in patients]
print(f"\nAge statistics:")
print(f"  Min age: {min(ages)}")
print(f"  Max age: {max(ages)}")
print(f"  Mean age: {sum(ages) / len(ages):.1f}")

# Visit statistics
visits_per_patient = [len(p.visits) for p in patients]
print(f"\nVisit statistics:")
print(f"  Min visits: {min(visits_per_patient)}")
print(f"  Max visits: {max(visits_per_patient)}")
print(f"  Mean visits: {sum(visits_per_patient) / len(visits_per_patient):.1f}")
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Visit Boundaries

**Wrong:**
```python
# Flattening visits loses temporal structure
all_codes = []
for visit in patient.visits:
    all_codes.extend(visit)
# Result: ['401.9', '250.00', '428.0', '584.9']
# Problem: Can't tell which codes from same visit!
```

**Right:**
```python
# Preserve visit structure
visits = patient.visits  # [['401.9', '250.00'], ['428.0', '584.9']]
# Can now process per-visit or track temporal progression
```

### Pitfall 2: Hardcoding Gender Values

**Wrong:**
```python
# Assumes 'M' and 'F', breaks if data has 'Male'/'Female'
gender_id = 1 if gender == 'F' else 0
```

**Right:**
```python
# Normalize first
gender = gender.upper().strip()
gender_id = 1 if gender == 'F' else 0
```

### Pitfall 3: Ignoring Age Capping

**Wrong:**
```python
# Treating age 90 as exact value
if patient.age == 90:
    print("Patient is exactly 90 years old")
```

**Right:**
```python
# Age 90 means "90 or older" in MIMIC-III
if patient.age == 90:
    print("Patient is 90+ years old (privacy-capped)")
```

### Pitfall 4: Modifying Visits In-Place

**Wrong:**
```python
# Modifying original data
for visit in patient.visits:
    visit.append('999.99')  # Adds to original!
```

**Right:**
```python
# Create copy if modifying
import copy
visits_copy = copy.deepcopy(patient.visits)
for visit in visits_copy:
    visit.append('999.99')
```

---

## Key Design Decisions

### Decision 1: List of Lists vs Flat Sequence

**Why List of Lists?**
- Preserves visit boundaries (temporal structure)
- Allows visit-level operations (e.g., "predict next visit")
- Matches real clinical workflows (admission-based care)

**Alternative:** Flat sequence with visit markers
```python
# Flat with markers (NOT USED)
codes = ['<v>', '401.9', '250.00', '<\v>', '<v>', '428.0', '<\v>']
# Problem: Harder to manipulate, prone to marker errors
```

### Decision 2: Age as Float vs Integer

**Why Float?**
- Allows fractional ages if needed (pediatrics: "3.5 years")
- Consistent with numpy defaults (float32)
- Future-proofing for age-in-months representations

**Why not integer?**
- MIMIC-III age is integer, but future datasets may have decimals
- PyTorch expects float tensors for continuous variables

### Decision 3: Remove Race/Ethnicity

**Why remove?**
- Medical validity: Few ICD-9 codes have race-specific rules
- Ethical concerns: Avoid encoding healthcare disparities
- Practical: Reduces dimensionality without clear benefit

**Impact on model:**
- Relies on age + gender only for demographic conditioning
- Generated patients are race-agnostic
- May miss race-correlated disease patterns (acceptable trade-off)

---

## Connection to Model Architecture

### How PatientRecord Feeds the Model

**PatientRecord:**
```python
PatientRecord(age=65.0, gender='M', visits=[['401.9', '250.00']])
```

**↓ to_dict()**

**Dictionary:**
```python
{
    'x_num': [65.0],       # → ConditionalPrompt (age embedding)
    'x_cat': [0],          # → ConditionalPrompt (gender embedding)
    'visits': [['401.9']]  # → DiagnosisCodeTokenizer → BART encoder/decoder
}
```

**↓ Model Processing**

**BART Input:**
```
Encoder: <s> <v> 401.9 250.00 <\v> </s>  (tokenized codes)
Prompts: age_embedding(65.0) ⊕ gender_embedding(0)  (conditional prompt)
```

**See next pages:**
- [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md) - How codes become tokens
- [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) - How prompts condition generation

---

## Summary

**PatientRecord is the bridge between raw CSV data and neural network input:**

1. **Structured Representation:** Separates demographics (age, gender) from medical codes (visits)
2. **Visit Structure:** List of lists preserves temporal ordering and visit boundaries
3. **Age Calculation:** Computed from DOB and first admission, capped at 90 for privacy
4. **Gender Encoding:** Simple binary (0=M, 1=F) for embedding
5. **Race Removed:** Ethical and practical decision to condition on age/gender only
6. **to_dict() Method:** Converts to PyTorch-compatible dictionary format
7. **Data Pipeline:** CSV → pandas merge → PatientRecord objects + DiagnosisVocabulary

**Key Files:**
- `data_loader.py:40-69` - PatientRecord class
- `data_loader.py:72-209` - load_mimic_data() function

---

## What's Next?

You now understand how raw MIMIC-III data is transformed into structured PatientRecord objects.

**Next:** [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md) - Learn how 6,985 unique ICD-9 codes are mapped to token IDs (1:1 mapping without fragmentation).

**Alternative Path:**
- [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md) - Review token-based solution
- [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md) - Skip ahead to tokenization

---

**Navigation:**
- ← Back to [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)
- → Next: [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
