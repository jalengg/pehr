# Generation

**Last Updated:** October 24, 2025

This document describes how to generate synthetic patients using the trained model.

## Generation Modes

**Two primary modes:**

1. **Conditional Generation** - Reconstruct patient from partial code prompts
2. **Zero-Prompt Generation** - Generate from demographics only (age, sex)

## 1. Conditional Generation

**File:** `generate.py:generate_patient_sequence_conditional()`

### Purpose

Reconstruct a patient's visits given:
- Demographics (age, sex)
- Visit structure (number of visits)
- Partial code prompts (some codes revealed, others masked)

### Function Signature

```python
def generate_patient_sequence_conditional(
    model: PromptBartWithDemographicPrediction,
    tokenizer: DiagnosisCodeTokenizer,
    target_patient: PatientRecord,
    device: torch.device,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    prompt_prob: float = 0.5,  # Fraction of codes to reveal
    max_codes_per_visit: int = 20
) -> dict
```

### Parameters

- `target_patient`: Real patient from test set (provides structure)
- `prompt_prob`: Probability each code is revealed (0.0 = zero-prompt, 1.0 = all codes revealed)
- `temperature`: Sampling temperature (lower = more conservative)
- `top_k`: Top-K sampling (keep top 40 most likely tokens)
- `top_p`: Nucleus sampling (cumulative probability threshold 0.9)

### Process

```python
# 1. Extract demographics and visit structure
age = target_patient.age
sex = target_patient.gender
num_visits = len(target_patient.visits)

# 2. For each visit, randomly select prompt codes
for visit in target_patient.visits:
    prompt_codes = []
    for code in visit:
        if random.random() < prompt_prob:
            prompt_codes.append(code)

# 3. Create encoder input with prompts and masks
encoder_input = []
for visit, prompts in zip(visits, prompt_codes_per_visit):
    encoder_input.extend([tokenizer.v_token] + prompts + [tokenizer.mask_token] + [tokenizer.v_end_token])

# 4. Generate with model
generated_ids = model.generate(
    input_ids=encoder_input_tensor,
    x_num=age_tensor,
    x_cat=sex_tensor,
    max_new_tokens=256,
    do_sample=True,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    no_repeat_ngram_size=1,  # Prevent duplicates
    eos_token_id=tokenizer.eos_token_id
)

# 5. Parse generated sequence into visits
generated_visits = parse_sequence_to_visits(generated_ids, tokenizer)
```

### Output

```python
{
    'generated_visits': List[List[str]],  # [[code1, code2], [code3, code4]]
    'target_visits': List[List[str]],     # Original patient visits
    'prompt_codes_per_visit': List[List[str]],  # Codes revealed as prompts
    'demographics': {'age': float, 'gender': str},
    'num_visits': int,
    'num_codes': int
}
```

### Example

```python
# Input
target_patient = PatientRecord(
    visits=[["401.9", "250.00", "585.9"], ["428.0", "584.9"]],
    age=65,
    gender=0
)
prompt_prob = 0.5  # Reveal ~50% of codes

# Possible prompt selection
visit_1_prompts = ["401.9"]  # 250.00, 585.9 masked
visit_2_prompts = []  # Both codes masked

# Encoder input
<s> <v> 401.9 <mask> <\v> <v> <mask> <\v> </s>

# Generated output (example)
generated_visits = [
    ["401.9", "250.00", "272.0"],   # Reconstructed + generated
    ["428.0", "584.9"]               # Reconstructed
]
```

## 2. Zero-Prompt Generation

**File:** `generate.py:generate_patient_from_demographics()`

### Purpose

Generate fully synthetic patient from demographics only:
- Sample random age and sex (or use provided)
- Generate plausible visit structure
- Generate plausible diagnosis codes

### Function Signature

```python
def generate_patient_from_demographics(
    model: PromptBartWithDemographicPrediction,
    tokenizer: DiagnosisCodeTokenizer,
    device: torch.device,
    age: Optional[float] = None,  # If None, sample from N(60, 20)
    sex: Optional[int] = None,    # If None, sample from Bernoulli(0.56)
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    max_sequence_length: int = 256
) -> dict
```

### Process

```python
# 1. Sample demographics if not provided
if age is None:
    age = np.clip(np.random.normal(60, 20), 0, 90)
if sex is None:
    sex = 0 if np.random.rand() < 0.56 else 1

# 2. Create minimal encoder input (just start token)
encoder_input = torch.tensor([[tokenizer.bos_token_id]])

# 3. Create decoder input (just BOS, no prompt codes)
decoder_input = torch.tensor([[tokenizer.bos_token_id]])

# 4. Generate full sequence
generated_ids = model.generate(
    input_ids=encoder_input,
    decoder_input_ids=decoder_input,
    x_num=age_tensor,
    x_cat=sex_tensor,
    max_new_tokens=max_sequence_length,
    do_sample=True,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    no_repeat_ngram_size=1,
    eos_token_id=tokenizer.convert_tokens_to_ids("<END>")
)

# 5. Parse into visits
generated_visits = parse_sequence_to_visits(generated_ids, tokenizer)
```

### Output

```python
{
    'generated_visits': List[List[str]],
    'demographics': {'age': float, 'sex': str},
    'num_visits': int,
    'num_codes': int
}
```

### Example

```python
# Input
age = 72.5
sex = 0  # Male

# Generated output (example)
generated_visits = [
    ["401.9", "250.00", "585.9", "272.0"],  # Visit 1: Hypertension, diabetes, CKD
    ["428.0", "584.9", "41401"]             # Visit 2: Heart failure, acute kidney injury
]
```

## 3. Sampling Parameters

### Temperature

**Range:** 0.1 - 2.0

- **Low (0.1-0.5):** Conservative, high-probability codes only
  - Pros: More realistic, coherent sequences
  - Cons: Less diverse, repetitive
- **Medium (0.6-0.8):** Balanced
  - Recommended for most use cases
- **High (0.9-2.0):** Exploratory, includes rare codes
  - Pros: More diverse
  - Cons: Less realistic, potentially incoherent

**Recommendation:** 0.3 for conditional, 0.7 for zero-prompt

### Top-K Sampling

**Range:** 10 - 100

- Keeps only top-K most likely tokens at each step
- **Low (10-20):** Very conservative
- **Medium (40-50):** Balanced (recommended)
- **High (80-100):** More diverse

**Recommendation:** 40

### Top-P (Nucleus) Sampling

**Range:** 0.7 - 1.0

- Keeps smallest set of tokens with cumulative probability ≥ p
- **0.7-0.8:** Conservative
- **0.9:** Balanced (recommended)
- **0.95-1.0:** Very diverse

**Recommendation:** 0.9

### No-Repeat N-gram Size

**Value:** 1 (always)

- Prevents immediate repetition of codes within a visit
- Essential for medical validity (can't diagnose same code twice in one visit)

## 4. Loading Trained Model

```python
from generate import load_trained_model

checkpoint_path = "checkpoints/best_model.pt"

model = load_trained_model(
    checkpoint_path=checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    device=device,
    logger=logger
)
```

## 5. Batch Generation

### Generate Multiple Patients

```python
# Conditional (reconstruct from test set)
test_patients = [...]  # Load test set

results = []
for patient in test_patients[:100]:
    result = generate_patient_sequence_conditional(
        model, tokenizer, patient, device, prompt_prob=0.0
    )
    results.append(result)

# Zero-prompt (fully synthetic)
results = []
for i in range(100):
    result = generate_patient_from_demographics(
        model, tokenizer, device
    )
    results.append(result)
```

## 6. Generation Scripts

### evaluate_medical_validity.py

**Purpose:** Generate 100 patients and evaluate medical validity

```bash
python evaluate_medical_validity.py
```

**Output:**
- Medical validity metrics (age/sex appropriateness, duplicates)
- Jaccard similarity (if conditional)

### evaluate_semantic_coherence.py

**Purpose:** Generate 100 patients and evaluate semantic coherence

```bash
python evaluate_semantic_coherence.py
```

**Output:**
- JS divergence (code frequency match)
- Distribution match (KS tests)
- Top-100 overlap
- Co-occurrence score

### test_unconditional.py

**Purpose:** Generate 10 patients for quick inspection

```bash
python test_unconditional.py
```

**Output:**
- Human-readable patient records
- Demographics and visit summaries

## 7. Common Issues

### Issue 1: Repetitive Codes

**Symptoms:** Same codes repeated across visits

**Solution:**
- Increase temperature (0.3 → 0.7)
- Increase top_k (40 → 80)
- Check `no_repeat_ngram_size=1` is set

### Issue 2: Too Many Codes Per Visit

**Symptoms:** 50+ codes in single visit (unrealistic)

**Solution:**
- Use `max_codes_per_visit=20` parameter
- Lower temperature
- Check model auxiliary loss weights

### Issue 3: Medical Invalidity

**Symptoms:** Pediatric codes for elderly, pregnancy codes for males

**Solution:**
- Increase auxiliary loss weights during training
- Apply post-processing filters (see docs/reference/medical_validity.md)

## Next Steps

- **Evaluate quality:** See [Evaluation](06_EVALUATION.md)
- **Understand model:** See [Model Architecture](03_MODEL_ARCHITECTURE.md)
- **Learn usage:** See [Usage Guide](07_USAGE_GUIDE.md)
