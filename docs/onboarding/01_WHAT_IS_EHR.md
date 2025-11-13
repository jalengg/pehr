# 01: What is Electronic Health Records (EHR)?

**Estimated Time:** 30 minutes
**Prerequisites:** None (start here!)
**Next:** [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)

---

## Learning Objectives

By the end of this page, you will understand:
- What electronic health records (EHRs) are and why they matter
- The structure of MIMIC-III, the dataset we use for training
- What ICD-9 diagnosis codes represent
- Why synthetic EHR generation is valuable
- Real-world applications of synthetic medical data

---

## What is an Electronic Health Record?

An **Electronic Health Record (EHR)** is a digital version of a patient's medical history, maintained by healthcare providers over time. Think of it as a comprehensive medical biography.

### Typical EHR Components

**Demographics:**
- Age, sex, race/ethnicity
- Insurance information
- Contact details

**Clinical Data:**
- Diagnoses (what conditions the patient has)
- Procedures (surgeries, tests performed)
- Medications (prescriptions, dosages)
- Lab results (blood work, imaging)
- Vital signs (blood pressure, heart rate, temperature)
- Clinical notes (doctor observations, nursing notes)

**Administrative Data:**
- Admission/discharge dates
- Hospital locations (ICU, general ward)
- Billing codes
- Provider information

### Key Characteristics

1. **Longitudinal:** Tracks patient history over time (months to years)
2. **Structured + Unstructured:** Coded data (diagnoses) + free text (notes)
3. **Multi-visit:** Multiple hospital admissions, outpatient visits
4. **High-dimensional:** Thousands of possible diagnosis codes, medications, procedures

---

## MIMIC-III: Our Training Dataset

**MIMIC-III** (Medical Information Mart for Intensive Care III) is a freely available critical care database developed by MIT.

### Dataset Overview

**Size:**
- 46,520 patients
- 58,976 hospital admissions
- 651,047 diagnosis records
- Collected: 2001-2012 at Beth Israel Deaconess Medical Center (Boston)

**Access:**
- Publicly available after CITI training completion
- De-identified: All PHI (Protected Health Information) removed
- Research use only

### MIMIC-III File Structure

For PromptEHR, we use three primary CSV files:

#### 1. PATIENTS.csv
**Purpose:** Patient demographics and mortality

**Key Columns:**
- `SUBJECT_ID`: Unique patient identifier (e.g., 10006)
- `GENDER`: M (male) or F (female)
- `DOB`: Date of birth (dates shifted for privacy)
- `DOD`: Date of death (if deceased)

**Example Row:**
```
SUBJECT_ID,GENDER,DOB,DOD
10006,M,2094-03-05,2165-05-15
```

**Our Usage:** Extract age and gender for demographic conditioning.

#### 2. ADMISSIONS.csv
**Purpose:** Hospital admission information

**Key Columns:**
- `HADM_ID`: Unique admission identifier
- `SUBJECT_ID`: Links to PATIENTS.csv
- `ADMITTIME`: Admission timestamp
- `DISCHTIME`: Discharge timestamp
- `ETHNICITY`: Patient ethnicity (often inconsistent)

**Example Row:**
```
HADM_ID,SUBJECT_ID,ADMITTIME,DISCHTIME,ETHNICITY
142345,10006,2165-03-14 07:15:00,2165-03-28 18:30:00,WHITE
```

**Our Usage:** Group diagnosis codes by admission to form "visits".

#### 3. DIAGNOSES_ICD.csv
**Purpose:** Diagnosis codes assigned during each admission

**Key Columns:**
- `HADM_ID`: Links to ADMISSIONS.csv
- `ICD9_CODE`: Diagnosis code (see next section)
- `SEQ_NUM`: Sequence number (1 = primary diagnosis)

**Example Row:**
```
HADM_ID,ICD9_CODE,SEQ_NUM
142345,4019,1
142345,25000,2
142345,5849,3
```

**Our Usage:** These ICD-9 codes are what our model learns to generate.

### Data Processing Pipeline

```
PATIENTS.csv + ADMISSIONS.csv + DIAGNOSES_ICD.csv
            ↓
    Join by SUBJECT_ID and HADM_ID
            ↓
    Group diagnoses by admission (visit)
            ↓
    Calculate age at admission
            ↓
    PatientRecord objects
```

**Implementation:** See `data_loader.py:100-250` (covered in [08_DATA_LOADING.md](08_DATA_LOADING.md))

---

## ICD-9 Diagnosis Codes

**ICD-9** (International Classification of Diseases, 9th Revision) is a medical coding system used to classify diseases and health conditions.

### Code Format

**Structure:** XXX.XX (3-5 characters)

**Examples:**
- `401.9` - Unspecified essential hypertension
- `250.00` - Diabetes mellitus without complication, type II
- `428.0` - Congestive heart failure
- `V58.61` - Long-term use of anticoagulants (V-codes are supplementary)
- `E849.0` - External cause: Home accident (E-codes are external causes)

### Hierarchy

ICD-9 codes have a natural hierarchy based on the first 3 digits:

**Category:** `401` (Hypertensive disease)
- `401.0` - Malignant hypertension
- `401.1` - Benign hypertension
- `401.9` - Unspecified hypertension

**Category:** `250` (Diabetes mellitus)
- `250.00` - Type II diabetes without complication
- `250.01` - Type I diabetes without complication
- `250.50` - Type II diabetes with ophthalmic manifestations

**This hierarchy is critical** - we use it for hierarchical generation (see [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)).

### Vocabulary Statistics (MIMIC-III)

After processing MIMIC-III, we extract:
- **6,985 unique ICD-9 diagnosis codes**
- **943 unique categories** (3-digit prefixes)
- **Average 7.4 codes per category**

**Distribution Characteristics:**
- **Long tail:** Few codes are very common (e.g., 401.9 appears in 15% of patients)
- **Sparse:** Most codes appear in <1% of patients
- **Co-occurrence patterns:** Certain codes frequently appear together (e.g., hypertension + heart failure)

---

## Why Synthetic EHR Generation?

### Problem: Real EHR Data Has Privacy Constraints

**HIPAA Regulations (US):**
- Protected Health Information (PHI) cannot be freely shared
- Includes: Names, dates, locations, medical record numbers
- De-identification is complex and may not be sufficient

**GDPR (EU):**
- Even stricter privacy requirements
- Patient consent required for data use

**Access Barriers:**
- Researchers need IRB approval (months-long process)
- Data use agreements with hospitals
- Secure computing environments (no data export)

### Solution: Generate Synthetic Data

**Synthetic EHR data** preserves statistical properties of real data while containing no actual patient information.

**Benefits:**
1. **No privacy concerns:** Fully artificial, freely shareable
2. **Unlimited scale:** Generate millions of patients if needed
3. **Controlled experiments:** Adjust demographics, prevalence
4. **Bias mitigation:** Balance underrepresented groups
5. **Software testing:** Realistic test data for EHR systems

### Applications

**1. Machine Learning Research**
- Train predictive models (readmission risk, mortality)
- Develop clinical decision support systems
- Test model fairness across demographics

**2. Software Development**
- Test EHR software without real patient data
- Generate edge cases for QA testing
- Performance benchmarking with large datasets

**3. Medical Education**
- Create realistic case studies for training
- Simulate patient populations for epidemiology courses
- Practice with rare conditions (hard to find real data)

**4. Clinical Trial Design**
- Simulate patient recruitment pools
- Estimate statistical power
- Design inclusion/exclusion criteria

**5. Health Policy Research**
- Model disease prevalence under different scenarios
- Evaluate interventions at population scale
- Study healthcare utilization patterns

---

## PromptEHR's Approach

PromptEHR generates synthetic patients with:

**Input (Demographics):**
- Age (continuous, e.g., 65.3 years)
- Sex (categorical, M or F)

**Output (Medical History):**
- Multiple hospital visits
- Each visit contains multiple ICD-9 diagnosis codes
- Realistic co-occurrence patterns
- Age/sex appropriate codes

**Example Generated Patient:**
```
Age: 67.2, Sex: M
Visit 1: [401.9, 250.00, 272.4]  # Hypertension, diabetes, hyperlipidemia
Visit 2: [428.0, 584.9, 401.9]   # Heart failure, kidney disease, hypertension
```

**Key Features:**
1. **Demographically conditioned:** Age/sex influence which codes appear
2. **Medically valid:** No pregnancy codes for males, no pediatric codes for elderly
3. **Semantically coherent:** Codes co-occur as in real data (e.g., hypertension + heart failure)
4. **Multi-visit structure:** Preserves longitudinal patient trajectory

---

## Challenges in Synthetic EHR Generation

### Challenge 1: High Dimensionality
- 6,985 possible diagnosis codes (compared to ~50K words in typical NLP)
- Extremely sparse: Most patients have 5-30 codes total
- Long-tail distribution: 80% of codes appear in <5% of patients

### Challenge 2: Medical Validity
- Age-appropriate: No pregnancy (630-679) for elderly males
- Sex-appropriate: No prostate cancer (600-608) for females
- Biologically plausible: No conflicting diagnoses

### Challenge 3: Semantic Coherence
- Codes must co-occur realistically
- Example: Hypertension (401.9) + Heart failure (428.0) are common together
- Example: Acne (706.1) + Heart failure (428.0) are unrealistic together
- Simple language models don't capture these patterns well

### Challenge 4: Sequential Structure
- Patients have multiple visits over time
- Chronic conditions persist (diabetes appears in later visits)
- Acute conditions resolve (infection may not recur)

**PromptEHR addresses these challenges through:**
- 1:1 token mapping (Challenge 1) - [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)
- Multi-task learning (Challenge 2) - [15_MULTITASK_LEARNING.md](15_MULTITASK_LEARNING.md)
- Co-occurrence regularization (Challenge 3) - [28_COOCCURRENCE_REGULARIZATION.md](28_COOCCURRENCE_REGULARIZATION.md)
- BART encoder-decoder (Challenge 4) - [11_BART_FOUNDATIONS.md](11_BART_FOUNDATIONS.md)

---

## Real-World Context: Why This Matters

### Case Study 1: COVID-19 Research
During the pandemic, researchers needed large-scale patient data to:
- Predict ICU admission risk
- Identify comorbidities associated with severe outcomes
- Design clinical trials

**Problem:** Privacy laws prevented rapid data sharing across hospitals.

**Synthetic Data Solution:** Generate synthetic COVID-19 patients with realistic comorbidity patterns (diabetes, hypertension, obesity) for initial model development.

### Case Study 2: Rare Disease Research
Genetic disorders (e.g., cystic fibrosis) affect few patients:
- Insufficient data for machine learning
- Cannot share data across institutions

**Synthetic Data Solution:** Generate synthetic rare disease patients by adjusting prevalence rates, enabling algorithm development.

### Case Study 3: Algorithmic Bias Testing
Predictive models may perform poorly for underrepresented groups (racial minorities, certain age groups).

**Synthetic Data Solution:** Generate balanced synthetic populations to test model fairness before clinical deployment.

---

## Try It Yourself

### Exercise 1: Explore MIMIC-III Files

```bash
# Navigate to data directory
cd /u/jalenj4/pehr_scratch/data_files

# Check file sizes
wc -l PATIENTS.csv ADMISSIONS.csv DIAGNOSES_ICD.csv

# View first 10 rows of PATIENTS.csv
head -10 PATIENTS.csv

# Count unique patients
tail -n +2 PATIENTS.csv | cut -d',' -f1 | sort -u | wc -l

# Count unique ICD-9 codes
tail -n +2 DIAGNOSES_ICD.csv | cut -d',' -f5 | sort -u | wc -l
```

**Expected Output:**
- PATIENTS.csv: 46,521 rows (46,520 patients + header)
- ADMISSIONS.csv: 58,977 rows (58,976 admissions + header)
- DIAGNOSES_ICD.csv: 651,048 rows (651,047 diagnoses + header)
- Unique ICD-9 codes: ~6,985

### Exercise 2: Look Up ICD-9 Codes

Use an online ICD-9 lookup tool (e.g., https://www.icd9data.com/) to understand these codes:

- `401.9` - ?
- `250.00` - ?
- `428.0` - ?
- `V58.61` - ?

**Answers:**
- `401.9` - Unspecified essential hypertension
- `250.00` - Diabetes mellitus without mention of complication, type II or unspecified type
- `428.0` - Congestive heart failure, unspecified
- `V58.61` - Long-term (current) use of anticoagulants

### Exercise 3: Conceptual Question

**Question:** Why can't we just use GPT-3 (pretrained on text) to generate ICD-9 codes directly?

**Hints:**
- How does GPT-3 tokenize "401.9"?
- Does GPT-3 know that "401.9" is a medical code vs a version number?

**Answer:** GPT-3 would fragment "401.9" into subwords like ["401", ".", "9"] and has no medical semantic understanding. It might generate syntactically valid but medically meaningless codes like "123.45" (doesn't exist in ICD-9).

---

## Key Takeaways

1. **EHRs are complex, multi-dimensional medical records** containing demographics, diagnoses, procedures, medications, and clinical notes.

2. **MIMIC-III is our training dataset**: 46,520 patients, 58,976 admissions, 651,047 diagnosis records.

3. **ICD-9 codes are structured medical codes** with format XXX.XX and natural hierarchy (943 categories, 6,985 specific codes).

4. **Synthetic EHR generation solves privacy problems** by creating artificial data with realistic statistical properties.

5. **Real-world applications include** ML research, software testing, medical education, clinical trials, and health policy.

6. **Key challenges are** high dimensionality, medical validity, semantic coherence, and sequential structure.

---

## What's Next?

You now understand **what** EHRs are and **why** synthetic generation matters.

**Next:** [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md) - Learn why medical codes require fundamentally different treatment than natural language text.

**Jump Ahead (if impatient):**
- [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md) - See what happens when you treat medical codes as text (spoiler: disaster)

---

**Navigation:**
- ← Back to [00_START_HERE.md](00_START_HERE.md)
- → Next: [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)
- ↑ Up to Phase 1 Overview
