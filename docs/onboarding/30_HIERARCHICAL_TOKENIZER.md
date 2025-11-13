# 30: Hierarchical Tokenizer - Dual Vocabulary (Categories + Codes)

**Estimated Time:** 45 minutes
**Prerequisites:** [29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)
**Next:** [31_HIERARCHICAL_DATASET.md](31_HIERARCHICAL_DATASET.md)

---

## Learning Objectives

- Understand HierarchicalDiagnosisTokenizer structure
- Learn dual vocabulary: 7 special + 943 categories + 6,985 codes = 7,935 tokens
- Understand token ID layout with hierarchical offset
- Learn category→code mapping during decoding

---

## Token ID Layout

```
Token IDs:
0-6:       Special tokens (<PAD>, <BOS>, <EOS>, <v>, <\v>, <END>, <mask>)
7-949:     Category tokens (943 categories)
950-7934:  Code tokens (6,985 codes)
```

**Example:**
```python
# Category "401" (hypertension)
category_idx = 0  # First category in vocabulary
token_id = 7 + 0 = 7

# Code "401.9" (hypertension NOS)
code_idx = 0  # First code in vocabulary
token_id = 7 + 943 + 0 = 950
```

---

## HierarchicalDiagnosisTokenizer

### Structure (hierarchical_tokenizer.py)

```python
class HierarchicalDiagnosisTokenizer:
    def __init__(self, hierarchy: ICD9Hierarchy):
        self.hierarchy = hierarchy
        self.code_offset = 7  # After special tokens
        self.vocab_size = 7 + len(hierarchy.categories) + len(hierarchy.vocabulary)
        # 7 + 943 + 6,985 = 7,935 total tokens
```

### Encoding Categories

```python
def encode_categories(self, categories: List[str]) -> List[int]:
    """Encode category strings to token IDs."""
    return [self.hierarchy.category2idx[cat] + self.code_offset
            for cat in categories]
```

[IN PROGRESS - Full implementation details]

**See:** hierarchical_tokenizer.py for complete implementation

---

**Navigation:** [← 28](28_CO_OCCURRENCE_REGULARIZATION.md) | [→ 31](31_HIERARCHICAL_DATASET.md) | [↑ Index](00_START_HERE.md)
