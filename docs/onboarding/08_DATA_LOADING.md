# 08: Data Loading - From MIMIC-III CSVs to PatientRecord Objects

**Estimated Time:** 60 minutes
**Prerequisites:** [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md), [06_VOCABULARY_CONSTRUCTION.md](06_VOCABULARY_CONSTRUCTION.md)
**Next:** [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)

---

## Learning Objectives

- Understand the load_mimic_data() function (data_loader.py:72-209)
- Master the pandas merge strategy for combining three CSV files
- Learn age calculation from DOB and first admission
- Understand visit grouping by HADM_ID (hospital admission ID)
- Learn chronological sorting by ADMITTIME and SEQ_NUM

---

## The Three MIMIC-III Files

### File Structure

**PATIENTS.csv** (46,520 rows)
```
SUBJECT_ID | GENDER | DOB
10006      | M      | 2094-03-05
10007      | F      | 2178-05-26
```

**ADMISSIONS.csv** (~58,000 rows - some patients have multiple admissions)
```
SUBJECT_ID | HADM_ID | ADMITTIME           | ETHNICITY
10006      | 142345  | 2180-07-02 15:45:00 | WHITE
10006      | 173289  | 2180-08-06 09:30:00 | WHITE
```

**DIAGNOSES_ICD.csv** (~651,000 rows - multiple diagnoses per admission)
```
SUBJECT_ID | HADM_ID | ICD9_CODE | SEQ_NUM
10006      | 142345  | 4019      | 1
10006      | 142345  | 25000     | 2
10006      | 142345  | 5859      | 3
10006      | 173289  | 4280      | 1
```

**Relationships:**
- One patient (SUBJECT_ID) → Many admissions (HADM_ID)
- One admission (HADM_ID) → Many diagnoses (ICD9_CODE)
- SEQ_NUM: Order of diagnoses (1 = primary, 2+ = secondary)

---

## The load_mimic_data() Function

### High-Level Algorithm (data_loader.py:72-209)

```python
def load_mimic_data(
    patients_path: str,
    admissions_path: str,
    diagnoses_path: str,
    logger: logging.Logger,
    num_patients: int = None
) -> Tuple[List[PatientRecord], DiagnosisVocabulary]:
```

**Steps:**
1. Load three CSV files into pandas DataFrames
2. Calculate age at first admission
3. Merge DataFrames (patients + admissions + diagnoses)
4. Sort chronologically (by patient, admission time, diagnosis sequence)
5. Group by SUBJECT_ID to process each patient
6. For each patient: Extract demographics, group visits by HADM_ID
7. Build PatientRecord objects and DiagnosisVocabulary simultaneously
8. Return patient list + vocabulary

---

## Step 1: Load CSV Files

### Implementation (data_loader.py:91-108)

```python
try:
    patients_df = pd.read_csv(patients_path, parse_dates=['DOB'])
    logger.info(f"Loaded {len(patients_df)} patients")

    admissions_df = pd.read_csv(admissions_path, parse_dates=['ADMITTIME'])
    logger.info(f"Loaded {len(admissions_df)} admissions")

    diagnoses_df = pd.read_csv(diagnoses_path)
    logger.info(f"Loaded {len(diagnoses_df)} diagnosis records")

except FileNotFoundError as e:
    logger.error(f"Required file not found: {e.filename}")
    return [], DiagnosisVocabulary()
except Exception as e:
    logger.error(f"Unexpected error during file loading: {e}")
    return [], DiagnosisVocabulary()
```

**Key points:**
- `parse_dates=['DOB']` → Converts DOB to datetime (for age calculation)
- `parse_dates=['ADMITTIME']` → Converts admission time to datetime (for sorting)
- ICD9_CODE stays as string (no need for date parsing)
- Error handling returns empty list + empty vocab (graceful failure)

---

## Step 2: Calculate Age at First Admission

### Implementation (data_loader.py:110-123)

```python
# Find first admission for each patient
first_admissions = admissions_df.loc[
    admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
][['SUBJECT_ID', 'ADMITTIME']]

# Merge with patient demographics
demo_df = pd.merge(
    patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
    first_admissions,
    on='SUBJECT_ID',
    how='inner'
)

# Calculate age (year difference)
demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
# Cap at 90 for HIPAA privacy
demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])
```

**Why first admission?**
- Consistent age reference across patients
- Model conditions on "patient at initial encounter"
- Later admissions may be years apart (age changes)

**Age capping (> 89 → 90):**
- MIMIC-III privacy requirement (HIPAA)
- Prevents re-identification of very elderly patients

---

## Step 3: Merge DataFrames

### Merge Strategy (data_loader.py:125-142)

**Step 3a: Merge admissions + diagnoses (by SUBJECT_ID + HADM_ID)**

```python
# Get admission info
admissions_info = admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY']]

# Merge with diagnoses
merged_df = pd.merge(
    admissions_info,
    diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']],
    on=['SUBJECT_ID', 'HADM_ID'],
    how='inner'
)
```

**Result:** Each row = one diagnosis code with admission info

**Step 3b: Merge with demographics (by SUBJECT_ID)**

```python
# Add demographics (age, gender)
final_df = pd.merge(
    merged_df,
    demo_df[['SUBJECT_ID', 'AGE', 'GENDER']],
    on='SUBJECT_ID',
    how='left'
)
```

**Final DataFrame columns:**
```
SUBJECT_ID | HADM_ID | ADMITTIME | ICD9_CODE | SEQ_NUM | AGE | GENDER
```

---

## Step 4: Sort Chronologically

### Implementation (data_loader.py:144-145)

```python
# Sort chronologically
final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)
```

**Three-level sort:**
1. **SUBJECT_ID:** Group all rows for same patient together
2. **ADMITTIME:** Within patient, sort visits by admission time
3. **SEQ_NUM:** Within visit, sort diagnoses by sequence number (1=primary, 2+=secondary)

**Why this order matters:**
- Model learns temporal patterns (disease progression)
- Primary diagnosis (SEQ_NUM=1) appears first in each visit
- Chronological order enables next-visit prediction

---

## Step 5: Group by Patient

### Implementation (data_loader.py:149-188)

```python
vocab = DiagnosisVocabulary()
patient_records = []

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

        # Add codes to vocabulary
        vocab.add_codes(icd_codes)

        visits.append(icd_codes)

    # Create patient record
    record = PatientRecord(
        subject_id=int(subject_id),
        age=age,
        gender=gender,
        visits=visits
    )
    patient_records.append(record)

    if num_patients is not None and len(patient_records) >= num_patients:
        break
```

**Key operations:**
- `groupby('SUBJECT_ID')` → Iterate over patients
- `groupby('HADM_ID', sort=False)` → Group diagnoses by visit (preserve chronological order)
- `astype(str)` → Convert ICD9_CODE to string (may be numeric in CSV)
- `vocab.add_codes(icd_codes)` → Build vocabulary during loading (efficient!)
- Early exit if `num_patients` limit reached (for testing with subsets)

---

## Step 6: Log Statistics

### Implementation (data_loader.py:189-208)

```python
logger.info(f"Loaded {len(patient_records)} patient records")
logger.info(f"Diagnosis vocabulary size: {len(vocab)}")

if len(patient_records) > 0:
    avg_visits = np.mean([len(r.visits) for r in patient_records])
    avg_codes_per_visit = np.mean([len(code_list) for r in patient_records
                                    for code_list in r.visits])

    logger.info(f"Average visits per patient: {avg_visits:.2f}")
    logger.info(f"Average codes per visit: {avg_codes_per_visit:.2f}")

    # Gender distribution
    gender_counts = pd.Series([r.gender for r in patient_records]).value_counts()
    logger.info(f"Gender distribution: {gender_counts.to_dict()}")

    # Sample record
    sample = patient_records[0]
    logger.debug(f"Sample patient: age={sample.age}, gender={sample.gender}")
    logger.debug(f"Sample visits: {sample.visits[:2]}")
```

**Typical output (25,000 patients):**
```
INFO: Loaded 25000 patient records
INFO: Diagnosis vocabulary size: 5562
INFO: Average visits per patient: 3.7
INFO: Average codes per visit: 8.2
INFO: Gender distribution: {'M': 14234, 'F': 10766}
DEBUG: Sample patient: age=86.0, gender=M
DEBUG: Sample visits: [['4019', '25000', '5859'], ['4280', '5849']]
```

---

## Visit Grouping: Why HADM_ID Matters

### Hospital Admission ID (HADM_ID)

**Definition:** Unique identifier for each hospital admission in MIMIC-III

**One patient, multiple admissions:**
```
SUBJECT_ID=10006:
  HADM_ID=142345 (July 2, 2180)  → Visit 1
  HADM_ID=173289 (Aug 6, 2180)   → Visit 2
```

**One admission, multiple diagnoses:**
```
HADM_ID=142345:
  ICD9_CODE=4019  (SEQ_NUM=1)  → Primary diagnosis
  ICD9_CODE=25000 (SEQ_NUM=2)  → Secondary diagnosis
  ICD9_CODE=5859  (SEQ_NUM=3)  → Secondary diagnosis
```

### Grouping Logic

**Without grouping (WRONG):**
```python
# Treats each diagnosis as separate visit
visits = [[code] for code in all_codes]
# Result: [["4019"], ["25000"], ["5859"], ["4280"], ...]
# Problem: Loses co-occurrence within admission!
```

**With HADM_ID grouping (RIGHT):**
```python
visit_groups = patient_data.groupby('HADM_ID', sort=False)
for _, visit_data in visit_groups:
    icd_codes = visit_data['ICD9_CODE'].tolist()
    visits.append(icd_codes)
# Result: [["4019", "25000", "5859"], ["4280", ...]]
# Correct: Preserves co-occurrence within admission
```

---

## Real Example: Patient 10006 Step-by-Step

[IN PROGRESS - Detailed walkthrough of single patient processing]

**Key points:**
- SUBJECT_ID=10006, age=86, gender=M
- Two admissions (HADM_ID: 142345, 173289)
- First admission: 3 codes, Second admission: 2 codes
- Final PatientRecord: visits=[["4019", "25000", "5859"], ["4280", "5849"]]

---

## Memory Optimization

[IN PROGRESS - Strategies for loading large datasets]

**Key points:**
- Load CSVs with `low_memory=False` option
- Use `num_patients` parameter to limit memory usage during development
- Full dataset (46K patients): ~2GB RAM
- Training subset (25K patients): ~1GB RAM

---

## Try It Yourself

### Exercise 1: Load Small Subset

```python
from data_loader import load_mimic_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load first 100 patients
patients, vocab = load_mimic_data(
    patients_path='data_files/PATIENTS.csv',
    admissions_path='data_files/ADMISSIONS.csv',
    diagnoses_path='data_files/DIAGNOSES_ICD.csv',
    logger=logger,
    num_patients=100
)

print(f"Loaded {len(patients)} patients")
print(f"Vocabulary size: {len(vocab)}")
print(f"\nFirst patient:")
print(f"  Age: {patients[0].age}")
print(f"  Gender: {patients[0].gender}")
print(f"  Visits: {len(patients[0].visits)}")
```

[IN PROGRESS - Additional exercises on visit grouping, age calculation, and statistics]

---

## Common Pitfalls

### Pitfall 1: Losing Chronological Order

**Wrong:**
```python
# Using default groupby (may reorder)
visit_groups = patient_data.groupby('HADM_ID')
```

**Right:**
```python
# Preserve chronological order with sort=False
visit_groups = patient_data.groupby('HADM_ID', sort=False)
```

### Pitfall 2: Ignoring SEQ_NUM

**Wrong:**
```python
# Grouping by HADM_ID without sorting by SEQ_NUM first
# Result: Diagnosis order within visit is random
```

**Right:**
```python
# Sort by SEQ_NUM before grouping (done in step 4)
final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)
```

[IN PROGRESS - Additional pitfalls]

---

## Summary

**load_mimic_data() transforms relational CSV data into structured PatientRecord objects:**

1. **Load CSVs:** Parse dates for age calculation and sorting
2. **Calculate Age:** Year difference from DOB to first admission, cap at 90
3. **Merge DataFrames:** Combine patients + admissions + diagnoses
4. **Sort Chronologically:** By SUBJECT_ID, ADMITTIME, SEQ_NUM
5. **Group by Patient:** Iterate over SUBJECT_ID groups
6. **Group by Visit:** Within patient, group diagnoses by HADM_ID
7. **Build Vocabulary:** Add codes during loading (efficient single pass)
8. **Create PatientRecords:** Structured objects with demographics + visits

**Key Files:**
- `data_loader.py:72-209` - load_mimic_data() function
- `data_loader.py:40-69` - PatientRecord class

---

## What's Next?

**Next:** [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md) - Learn BART-style denoising with span masking, token deletion, and replacement

**Alternative:**
- [10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md) - Skip to end-to-end pipeline
- [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) - Jump to model architecture

---

**Navigation:**
- ← Back to [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)
- → Next: [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
