# 31: Hierarchical Dataset - Category Sequence Training Data

**Estimated Time:** 45 minutes
**Prerequisites:** [30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)
**Next:** [32_HIERARCHICAL_TRAINING.md](32_HIERARCHICAL_TRAINING.md)

---

## Learning Objectives

- Understand HierarchicalEHRDataset structure
- Learn code→category conversion during data loading
- Understand corruption applied to category sequences
- Learn how hierarchy improves data augmentation

---

## Code→Category Conversion

### Example

**Original patient (code level):**
```python
PatientRecord(
    visits=[
        ["401.9", "250.00", "585.9"],  # Visit 1
        ["428.0", "584.9"]              # Visit 2
    ]
)
```

**Converted to categories:**
```python
category_visits = [
    ["401", "250", "585"],  # Visit 1 (categories)
    ["428", "584"]          # Visit 2 (categories)
]
```

### Implementation

```python
class HierarchicalEHRDataset(Dataset):
    def __getitem__(self, idx):
        patient = self.patient_records[idx]

        # Convert codes to categories
        category_visits = []
        for visit in patient.visits:
            categories = [self.hierarchy.code_to_category[code]
                          for code in visit]
            # Remove duplicate categories within visit
            categories = list(dict.fromkeys(categories))
            category_visits.append(categories)

        # Apply corruption (same as flat dataset)
        corrupted_visits = self.corruptor.corrupt_sequence(category_visits)

        # Tokenize category sequences
        input_ids = self.tokenizer.encode_patient(corrupted_visits, ...)
        return {'input_ids': input_ids, 'x_num': age, 'x_cat': gender}
```

[IN PROGRESS - Full dataset implementation]

**See:** hierarchical_dataset.py for complete implementation

---

**Navigation:** [← 30](30_HIERARCHICAL_TOKENIZER.md) | [→ 32](32_HIERARCHICAL_TRAINING.md) | [↑ Index](00_START_HERE.md)
