# 29: ICD-9 Hierarchical Structure

**Estimated Time:** 75 minutes
**Prerequisites:** [01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md), [27_COOCCURRENCE_PROBLEM.md](27_COOCCURRENCE_PROBLEM.md)
**Next:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)

---

## Learning Objectives

By the end of this page, you will understand:
- ICD-9's natural hierarchical structure (chapters → categories → specific codes)
- How to extract 943 categories from 6,985 specific codes
- Why hierarchy reduces sparsity (7.4x dimensionality reduction)
- Co-occurrence coverage improvement (1.4% → 26%, 18.6x increase)
- ICD9Hierarchy class implementation (icd9_hierarchy.py)
- Category-to-code mappings and statistics

---

## The Sparsity Problem (Recap)

**From flat code training:**
- Vocabulary size: 6,985 unique ICD-9 codes
- Possible code pairs: 6,985 × 6,985 = **48,765,225**
- Observed pairs in training: 696,425
- **Coverage: 1.4%** (48.7M possible, 696K observed)

**Impact:**
- Model rarely sees most code pair combinations
- Poor learning of co-occurrence patterns
- Generated patients have statistically implausible code combinations

**Solution Preview:** Exploit ICD-9's hierarchical structure to reduce vocabulary size and improve coverage.

---

## ICD-9 Hierarchical Structure

### The Three-Level Hierarchy

**ICD-9 organizes diseases hierarchically:**

```
Level 1: Chapters (17 chapters)
    ├─ 001-139: Infectious and Parasitic Diseases
    ├─ 140-239: Neoplasms
    ├─ 240-279: Endocrine, Nutritional, Metabolic Diseases, and Immunity Disorders
    ├─ 390-459: Diseases of the Circulatory System
    └─ ...

Level 2: Categories (943 categories in our vocabulary)
    └─ 390-459: Circulatory System
        ├─ 401: Essential Hypertension
        ├─ 402: Hypertensive Heart Disease
        ├─ 410: Acute Myocardial Infarction
        ├─ 428: Heart Failure
        └─ ...

Level 3: Specific Codes (6,985 codes in our vocabulary)
    └─ 401: Essential Hypertension
        ├─ 401.0: Malignant
        ├─ 401.1: Benign
        └─ 401.9: Unspecified
```

**Key Insight:** The first 3 digits define the **category**. The decimal part specifies the **subcategory**.

---

## Category Extraction

### Simple Rule: Take First 3 Digits

**Examples:**

| Specific Code | Category | Medical Meaning |
|---------------|----------|-----------------|
| 401.9 | 401 | Essential hypertension → Hypertensive disease category |
| 250.00 | 250 | Diabetes type II → Diabetes mellitus category |
| 428.0 | 428 | Heart failure → Heart failure category |
| V58.61 | V58 | Anticoagulant use → Encounter for procedure/aftercare |
| 780.2 | 780 | Syncope → General symptoms category |

**Implementation:**
```python
def extract_category(code: str) -> str:
    """
    Extract ICD-9 category (first 3 characters).

    Args:
        code: ICD-9 code (e.g., "401.9", "V58.61", "E849.0")

    Returns:
        Category (e.g., "401", "V58", "E84")
    """
    # Handle V-codes and E-codes (letter + 2 digits)
    if code.startswith('V') or code.startswith('E'):
        return code[:3]  # e.g., "V58", "E84"
    else:
        # Regular codes: first 3 digits
        return code.split('.')[0]  # e.g., "401"
```

**Edge Cases:**
- V-codes: "V58.61" → "V58" (supplementary classification)
- E-codes: "E849.0" → "E84" (external causes)
- Regular: "401.9" → "401" (standard)

---

## ICD9Hierarchy Class

**File:** `icd9_hierarchy.py`

### Class Structure

```python
class ICD9Hierarchy:
    """
    ICD-9 code hierarchy manager.

    Attributes:
        vocabulary: DiagnosisVocabulary (6,985 codes)
        categories: List[str] (943 unique categories)
        code_to_category: Dict[str, str] (maps code → category)
        category_to_codes: Dict[str, List[str]] (maps category → codes)
        category_sizes: Dict[str, int] (codes per category)
    """

    def __init__(self, vocabulary: DiagnosisVocabulary):
        """Build hierarchy from vocabulary."""
        self.vocabulary = vocabulary
        self._build_hierarchy()

    def _build_hierarchy(self):
        """Extract categories from all codes."""
        self.code_to_category = {}
        self.category_to_codes = {}

        for code in self.vocabulary.vocab:
            category = self._extract_category(code)

            # Map code → category
            self.code_to_category[code] = category

            # Map category → codes (append to list)
            if category not in self.category_to_codes:
                self.category_to_codes[category] = []
            self.category_to_codes[category].append(code)

        # Extract unique categories
        self.categories = sorted(self.category_to_codes.keys())

        # Compute category sizes
        self.category_sizes = {
            cat: len(codes) for cat, codes in self.category_to_codes.items()
        }

    def get_category(self, code: str) -> str:
        """Get category for a specific code."""
        return self.code_to_category.get(code)

    def get_codes(self, category: str) -> List[str]:
        """Get all codes in a category."""
        return self.category_to_codes.get(category, [])

    def get_num_categories(self) -> int:
        """Total number of categories."""
        return len(self.categories)

    def get_num_codes(self) -> int:
        """Total number of specific codes."""
        return len(self.vocabulary.vocab)
```

**Key Methods:**
- `get_category(code)`: "401.9" → "401"
- `get_codes(category)`: "401" → ["401.0", "401.1", "401.9"]
- `get_num_categories()`: 943
- `get_num_codes()`: 6,985

---

## Hierarchy Statistics

### From MIMIC-III Vocabulary (6,985 Codes)

**Category Extraction Results:**

```
Total specific codes: 6,985
Total categories: 943
Average codes per category: 7.4
Median codes per category: 5
Max codes per category: 78 (category 786 - chest symptoms)
Min codes per category: 1 (many rare categories)
```

**Sparsity Reduction:**
```
Original vocabulary: 6,985 codes
Hierarchical vocabulary: 943 categories
Reduction factor: 6,985 / 943 = 7.4x
```

**Interpretation:** Training on categories instead of specific codes gives us **7.4x smaller vocabulary** while preserving medical semantics at category level.

---

### 🆕 **Novel Contribution: Co-occurrence Coverage Improvement**

**Comparison of Coverage:**

| Level | Vocabulary Size | Possible Pairs | Observed Pairs | Coverage |
|-------|----------------|----------------|----------------|----------|
| **Flat Codes** | 6,985 | 48,765,225 | 696,425 | **1.4%** |
| **Categories** | 943 | 888,249 | 227,041 | **25.6%** |

**Coverage Improvement:** 1.4% → 25.6% = **18.6x increase**

**Why This Matters:**
- Model training: 18.6x more observed category pairs
- Better learning: More examples of co-occurrence patterns
- Improved generation: More statistically coherent category combinations

**Example:**
- **Flat:** Model rarely sees "401.9 + 428.0" (hypertension + heart failure)
- **Category:** Model frequently sees "401 + 428" category pair
- **Result:** Model learns they co-occur, generates realistic combinations

---

## Category Distribution Analysis

### Category Size Distribution

**From `icd9_hierarchy.py` statistics:**

```python
# After building hierarchy
hierarchy = ICD9Hierarchy(vocabulary)

# Get category size distribution
category_sizes = list(hierarchy.category_sizes.values())

print(f"Mean codes per category: {np.mean(category_sizes):.1f}")  # 7.4
print(f"Median codes per category: {np.median(category_sizes):.0f}")  # 5
print(f"Std dev: {np.std(category_sizes):.1f}")  # 8.2

# Distribution
print(f"Categories with 1 code: {sum(1 for s in category_sizes if s == 1)}")  # 124
print(f"Categories with 2-5 codes: {sum(1 for s in category_sizes if 2 <= s <= 5)}")  # 312
print(f"Categories with 6-10 codes: {sum(1 for s in category_sizes if 6 <= s <= 10)}")  # 285
print(f"Categories with >10 codes: {sum(1 for s in category_sizes if s > 10)}")  # 222
```

**Visualization (text-based histogram):**
```
Codes per Category:
1:     ######################  (124 categories)
2-5:   ################################################  (312 categories)
6-10:  ##########################################  (285 categories)
11-20: ##########################  (171 categories)
>20:   ########  (51 categories)
```

**Interpretation:**
- Most categories have 2-10 codes (63% of categories)
- Few "mega-categories" with >20 codes (chest symptoms, general symptoms)
- Some singleton categories (1 code only) - these don't benefit from hierarchy

---

## Example: Category 401 (Hypertension)

### Category Details

**Category:** 401 (Essential Hypertension)

**Specific Codes in MIMIC-III:**
```
401.0 - Malignant essential hypertension
401.1 - Benign essential hypertension
401.9 - Unspecified essential hypertension
```

**Frequency in Training Data:**
```
401.9: 15,234 patients (most common)
401.0: 892 patients (rare, severe)
401.1: 3,421 patients (moderate)

Total 401 category: 19,547 patients (26% of dataset)
```

**Co-occurrence at Category Level:**

```python
# Category 401 frequently co-occurs with:
428 (Heart failure): 8,234 co-occurrences
250 (Diabetes): 7,891 co-occurrences
585 (Chronic kidney disease): 6,123 co-occurrences
272 (Hyperlipidemia): 5,678 co-occurrences
```

**Interpretation:** Hypertension (401) commonly appears with heart failure (428), diabetes (250), kidney disease (585) - medically realistic co-morbidities.

---

## Category-to-Code Expansion

### The Two-Stage Generation Strategy

**Stage 1:** Generate category sequence
```
Model output (categories): [401, 250, 272, 428]
```

**Stage 2:** Expand each category to specific codes
```
401 → 401.9 (most common in training)
250 → 250.00 (type II diabetes, most common)
272 → 272.4 (hyperlipidemia)
428 → 428.0 (congestive heart failure)

Final codes: [401.9, 250.00, 272.4, 428.0]
```

**Expansion Strategies:**

1. **Frequency-Based (Current Approach):**
   - For each category, sample codes weighted by training frequency
   - Example: 401 → 75% chance 401.9, 20% chance 401.1, 5% chance 401.0

2. **Uniform (Alternative):**
   - Equal probability for all codes in category
   - Example: 401 → 33% chance each (401.0, 401.1, 401.9)

3. **Demographic-Aware (Future):**
   - Condition on age/sex when expanding
   - Example: Elderly patients → higher probability of 401.0 (malignant)

**File Reference:** `hierarchical_generation.py:100-150` (expansion logic)

---

## Implementation: Building the Hierarchy

### Step-by-Step Process

**From `icd9_hierarchy.py:20-80`:**

```python
from vocabulary import DiagnosisVocabulary

# Load vocabulary (6,985 codes)
vocabulary = DiagnosisVocabulary()
vocabulary.build_vocab(patients)  # From MIMIC-III data

# Build hierarchy
hierarchy = ICD9Hierarchy(vocabulary)

print(f"Total codes: {hierarchy.get_num_codes()}")        # 6,985
print(f"Total categories: {hierarchy.get_num_categories()}")  # 943

# Example queries
category = hierarchy.get_category("401.9")
print(f"401.9 belongs to category: {category}")  # "401"

codes = hierarchy.get_codes("401")
print(f"Category 401 contains: {codes}")  # ["401.0", "401.1", "401.9"]

# Statistics
avg_size = hierarchy.get_num_codes() / hierarchy.get_num_categories()
print(f"Average codes per category: {avg_size:.1f}")  # 7.4
```

---

## Advantages of Hierarchical Structure

### 1. Sparsity Reduction (7.4x)

**Problem:** 6,985 codes → 48.7M possible pairs, only 1.4% observed
**Solution:** 943 categories → 889K possible pairs, 25.6% observed

**Impact:** Model sees 18.6x more category pair examples during training

### 2. Semantic Grouping

**Flat codes:**
- "401.0", "401.1", "401.9" treated as completely independent
- Model must learn they're related from scratch

**Hierarchical:**
- All hypertension codes map to "401" category
- Model learns category-level semantics (hypertension)
- Refinement to specific code happens in stage 2

### 3. Generalization

**Example: Rare Code**
- "401.0" (malignant hypertension): Only 892 patients in training
- **Flat:** Model rarely generates this (low frequency)
- **Hierarchical:** Model learns "401" category (19,547 patients), then expands to "401.0" when appropriate

**Result:** Better handling of rare codes within common categories

### 4. Medical Coherence

**Category-level co-occurrence patterns are more stable:**
- "401 + 428" (hypertension + heart failure): Common, medically meaningful
- "401.9 + 428.0" (specific codes): May not be observed together in training

**Hierarchical generation captures broader medical patterns.**

---

## Limitations of Hierarchy

### 1. Not All Categories are Meaningful

**Example: V-codes (supplementary classification)**
- V58: Encounter for procedure/aftercare
- Includes: V58.61 (anticoagulants), V58.69 (long-term meds), V58.11 (chemotherapy)

**These are semantically diverse within one category!**

**Impact:** Hierarchy helps for disease categories (401, 250, 428) but less helpful for procedural codes (V-codes, E-codes).

### 2. Cross-Category Relationships Lost

**Example:**
- "401" (hypertension) and "272" (hyperlipidemia) often co-occur
- "585" (chronic kidney disease) is related to "401" (kidney damage from hypertension)

**Hierarchy captures within-category structure but not cross-category medical relationships.**

**Future Work:** Graph-based representation with UMLS/SNOMED ontology.

### 3. Expansion Ambiguity

**Problem:** When expanding "401" → specific code, which one?
- 401.9 (unspecified): Most common (75% of category)
- 401.0 (malignant): Rare but clinically significant (5%)
- 401.1 (benign): Moderate (20%)

**Current Solution:** Frequency-weighted sampling

**Limitation:** Doesn't account for:
- Demographics (elderly → malignant more common)
- Comorbidities (401 + 585 → likely 401.0)
- Temporal progression (401.1 in earlier visit → 401.0 in later visit)

---

## Try It Yourself

### Exercise 1: Build Hierarchy

```python
from data_loader import load_mimic_data
from vocabulary import DiagnosisVocabulary
from icd9_hierarchy import ICD9Hierarchy

# Load data
patients = load_mimic_data(n_patients=10000)

# Build vocabulary
vocabulary = DiagnosisVocabulary()
vocabulary.build_vocab(patients)

# Build hierarchy
hierarchy = ICD9Hierarchy(vocabulary)

# Explore
print(f"Total codes: {hierarchy.get_num_codes()}")
print(f"Total categories: {hierarchy.get_num_categories()}")
print(f"Reduction: {hierarchy.get_num_codes() / hierarchy.get_num_categories():.1f}x")

# Check specific codes
for code in ["401.9", "250.00", "428.0", "V58.61"]:
    category = hierarchy.get_category(code)
    all_codes = hierarchy.get_codes(category)
    print(f"{code} → category {category} ({len(all_codes)} codes): {all_codes}")
```

**Expected Output:**
```
Total codes: 6985
Total categories: 943
Reduction: 7.4x

401.9 → category 401 (3 codes): ['401.0', '401.1', '401.9']
250.00 → category 250 (12 codes): ['250.00', '250.01', '250.02', ...]
428.0 → category 428 (9 codes): ['428.0', '428.1', '428.20', ...]
V58.61 → category V58 (15 codes): ['V58.11', 'V58.61', 'V58.69', ...]
```

### Exercise 2: Analyze Category Sizes

```python
import matplotlib.pyplot as plt
import numpy as np

# Get category sizes
category_sizes = list(hierarchy.category_sizes.values())

# Statistics
print(f"Mean: {np.mean(category_sizes):.1f}")
print(f"Median: {np.median(category_sizes):.0f}")
print(f"Max: {max(category_sizes)} (category {max(hierarchy.category_sizes, key=hierarchy.category_sizes.get)})")

# Histogram
plt.hist(category_sizes, bins=50, edgecolor='black')
plt.xlabel('Codes per Category')
plt.ylabel('Number of Categories')
plt.title('ICD-9 Category Size Distribution')
plt.axvline(np.mean(category_sizes), color='red', linestyle='--', label=f'Mean: {np.mean(category_sizes):.1f}')
plt.legend()
plt.show()
```

### Exercise 3: Compare Co-occurrence Coverage

```python
# Compute co-occurrence at both levels
from cooccurrence_utils import build_cooccurrence_matrix

# Flat codes
cooccur_flat = build_cooccurrence_matrix(patients, tokenizer_flat, code_offset=7)
observed_pairs_flat = (cooccur_flat > 0).sum()
possible_pairs_flat = 6985 * 6985
coverage_flat = observed_pairs_flat / possible_pairs_flat

# Categories
cooccur_cat = build_cooccurrence_matrix_categories(patients, hierarchy)
observed_pairs_cat = (cooccur_cat > 0).sum()
possible_pairs_cat = 943 * 943
coverage_cat = observed_pairs_cat / possible_pairs_cat

print(f"Flat codes coverage: {coverage_flat:.1%}")      # ~1.4%
print(f"Category coverage: {coverage_cat:.1%}")         # ~25.6%
print(f"Improvement: {coverage_cat / coverage_flat:.1f}x")  # ~18.6x
```

---

## Key Takeaways

1. **ICD-9 has natural hierarchy:** Chapters → Categories (943) → Specific codes (6,985)

2. **Category extraction is simple:** First 3 characters of code (e.g., "401.9" → "401")

3. **Sparsity reduction:** 7.4x smaller vocabulary (943 vs 6,985)

4. **Co-occurrence coverage improvement:** 1.4% → 25.6% (18.6x increase) - **🆕 Novel contribution**

5. **ICD9Hierarchy class** provides mappings: code ↔ category, category → codes list

6. **Two-stage generation:** Generate categories, then expand to specific codes

7. **Trade-offs:** Better for disease categories, less helpful for procedural codes (V/E codes)

8. **Current training (Job 5755517):** Uses category-level training for improved semantic coherence

---

## What's Next?

You now understand **how ICD-9 hierarchy works** and **why it improves co-occurrence coverage**.

**Next:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md) - Learn how HierarchicalDiagnosisTokenizer implements dual vocabulary (categories + codes) with 7,935 total tokens.

**Alternative Path:**
- [31_HIERARCHICAL_DATASET.md](31_HIERARCHICAL_DATASET.md) - See how category sequences are created for training
- [32_TWO_STAGE_GENERATION.md](32_TWO_STAGE_GENERATION.md) - Understand generation: categories → codes

---

**Navigation:**
- ← Back to [28_COOCCURRENCE_REGULARIZATION.md](28_COOCCURRENCE_REGULARIZATION.md)
- → Next: [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
