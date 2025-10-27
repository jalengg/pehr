# Conditional Reconstruction Analysis - Clinical Perspective

**Model**: PromptEHR (10k patients, 50 epochs)
**Test Set**: 2000 held-out patients
**Evaluation**: 20 random samples
**Average Jaccard**: 0.403 (40.3% code overlap)

---

## Clinical Examples with Translated ICD-9 Codes

### Example 1: Newborn Patient (Jaccard 0.200)

**Patient**: 0-year-old Female, White
**Target Conditions**:
- V3000: Single liveborn, born in hospital
- V053: Need for prophylactic vaccination - viral hepatitis
- V290: Observation for suspected infectious condition

**Given to Model** (50% prompt):
- V3000: Single liveborn, born in hospital ✓
- V290: Observation for suspected infectious condition ✓

**Model Generated**:
- V3000: Single liveborn, born in hospital ✓ (correct)
- 3572: Polyneuropathy in diabetes ❌ (inappropriate for newborn)
- 42979: Atrial fibrillation ❌ (inappropriate for newborn)

**Analysis**: Model correctly reconstructed 1/3 codes but hallucinated adult conditions for a newborn. Demonstrates **age-inappropriate generation** issue.

---

### Example 2: Cardiac Patient with Kidney Transplant (Jaccard 0.500)

**Patient**: 56-year-old Male, White
**Target Conditions**:
- 41401: Coronary atherosclerosis
- 9982: Tracheostomy status
- 2765: Hypokalemia
- V420: Kidney replaced by transplant
- 4582: Hypotension
- 42789: Cardiac dysrhythmias
- 2859: Anemia
- 412: Old myocardial infarction
- 4019: Hypertension

**Given to Model** (56% prompt):
- 9982: Tracheostomy status ✓
- V420: Kidney replaced by transplant ✓
- 42789: Cardiac dysrhythmias ✓
- 2859: Anemia ✓
- 4019: Hypertension ✓

**Model Correctly Inferred**:
- All 5 prompted codes ✓

**Model Incorrectly Generated**:
- 42979: Atrial fibrillation (added, but plausible for cardiac patient)
- **Issue**: Generated atrial fibrillation 4 times (duplicates)

**Missing**:
- 41401: Coronary atherosclerosis ❌
- 2765: Hypokalemia ❌
- 4582: Hypotension ❌
- 412: Old myocardial infarction ❌

**Analysis**: Model shows **good pattern matching** - recognized this is a cardiac patient and added cardiac codes. However, it **missed specific conditions** like coronary atherosclerosis and MI. The **duplicate generation** (42979 x4) is a technical issue.

---

### Example 3: Premature Twin (Jaccard 0.556)

**Patient**: 0-year-old Female, White
**Target Conditions**:
- V3101: Twin birth, mate liveborn, born in hospital
- 769: Respiratory distress syndrome in newborn
- 7461: Transitory tachypnea of newborn
- 7454: Abnormal weight loss
- 76515: Extreme immaturity, 1000-1249 grams
- 76525: Other preterm infants, 1000-1249 grams
- V290: Observation for suspected infectious condition

**Given to Model** (71% prompt):
- V3101: Twin birth ✓
- 7461: Transitory tachypnea ✓
- 7454: Abnormal weight loss ✓
- 76525: Preterm infant 1000-1249g ✓
- V290: Observation for infection ✓

**Model Correctly Maintained**:
- All 5 prompted codes ✓

**Model Incorrectly Generated**:
- 34690: Migraine ❌ (inappropriate for newborn)
- 42979: Atrial fibrillation ❌ (inappropriate for newborn)

**Missing**:
- 769: Respiratory distress syndrome ❌
- 76515: Extreme immaturity 1000-1249g ❌

**Analysis**: High prompt ratio (71%) led to good reconstruction, but model still added inappropriate adult conditions. **Missing respiratory distress** is clinically significant - it's common in premature twins.

---

## Key Patterns Observed

### 1. Code Frequency Bias
**Code 42979 (Atrial Fibrillation)** appears in almost every generation, including:
- Newborns (medically impossible)
- Young patients (rare)
- Non-cardiac cases

**Hypothesis**: This is one of the most common codes in the training data, so the model defaults to it when uncertain.

### 2. Duplicate Generation Issue
Many patients have the same code generated multiple times:
- Patient 2: 42979 appears 4 times
- Patient 5: 42979 appears 4 times

**Technical Issue**: The generation process doesn't properly suppress duplicates during sampling.

### 3. Age-Inappropriate Codes
Model frequently generates:
- Adult cardiac conditions for newborns
- Diabetes neuropathy for infants
- Migraines for neonates

**Root Cause**: Demographics (age) not strongly enough influencing code selection.

### 4. Missing Rare/Specific Codes
Model tends to miss:
- Specific cardiac conditions (coronary atherosclerosis, MI)
- Respiratory conditions in premature infants
- Rare complications

**Pattern**: Model prefers common codes over specific ones.

---

## Comparison: 3k vs 10k Training

| Metric | 3k Training | 10k Training | Change |
|--------|-------------|--------------|--------|
| **Jaccard** | 0.249 | 0.403 | +61.8% ✅ |
| **Age-appropriate codes** | Poor | Poor | No change ❌ |
| **Duplicate suppression** | Some issues | Major issues | Worse ❌ |
| **Code diversity** | Limited | Limited | No change |

**Key Finding**: More data improved **overall accuracy** but didn't fix **medical validity** issues.

---

## Clinical Plausibility Assessment

### Strengths
✅ **Comorbidity associations**: Model learns that cardiac patients have multiple cardiac conditions
✅ **Demographic patterns**: Recognizes newborns vs adults (mostly)
✅ **Code count accuracy**: Perfect match to target visit sizes

### Weaknesses
❌ **Age constraints**: Generates atrial fibrillation for newborns
❌ **Code specificity**: Misses specific diagnoses (MI, coronary atherosclerosis)
❌ **Duplicate suppression**: Same code generated multiple times
❌ **Rare code generation**: Defaults to common codes

---

## Recommendations for Improvement

### 1. Hard Age Constraints (Post-Processing)
Filter generated codes by age appropriateness:
- Age 0-1: Only neonatal/pediatric codes allowed
- Age 65+: Geriatric conditions more likely
- Certain codes impossible at certain ages

### 2. Duplicate Suppression During Generation
Modify sampling to:
- Track already-generated codes
- Suppress their logits to force diversity
- Use set-based deduplication in post-processing

### 3. Reweight Loss by Code Frequency
Give higher loss weight to:
- Rare codes (learn them better)
- Specific diagnoses (vs generic ones)
- Code combinations (vs individual codes)

### 4. Multi-Task Learning with Age Prediction
Add auxiliary task:
- Predict patient age from codes
- Force model to learn age-code associations
- Improve demographic conditioning

### 5. Add Code Co-occurrence Constraints
Use medical knowledge:
- Certain code combinations are impossible
- Others are highly correlated
- Encode as soft constraints in generation

---

## Conclusion

The 10k/50epoch model achieved **40.3% Jaccard** - a significant improvement from 24.9%. However, translation to human-readable descriptions reveals:

**Technical Success**:
- Good reconstruction accuracy
- Strong pattern learning
- Perfect code count control

**Medical Validity Concerns**:
- Age-inappropriate codes common
- Excessive duplicate generation
- Missing rare/specific conditions

**Next Steps**: See **MULTITASK_LEARNING.md** for comprehensive solution using auxiliary prediction tasks.

---

## Root Cause Analysis

### Why Medical Validity Issues Occur

**Problem:** Model generates age-inappropriate codes (atrial fibrillation in newborns) and duplicate codes (42979 appearing 4x).

**Root Cause #1: One-Way Conditioning**

Current architecture learns only:
```
P(codes | age, sex, race)  ← Forward direction
```

The loss function is:
```
Loss = CrossEntropyLoss(generated_codes, target_codes)
```

This loss **only cares about reconstruction accuracy**. It has **zero gradient signal** preventing age-inappropriate codes.

**Example failure:**
```python
Training: age=65, codes=[42979, 41401, 4019]  ← Model learns "42979 is common"
Generation: age=0 → Model generates [42979, ...]  ← No penalty for age violation!
```

**Root Cause #2: Manual Sampling Loop**

Current generation (generate.py:182-223) uses manual temperature/top-k/top-p sampling without duplicate suppression. Missing `no_repeat_ngram_size=1` parameter that HuggingFace's `model.generate()` provides.

### Solution: Multi-Task Learning

Add auxiliary tasks that force bidirectional consistency:
```
P(codes | age, sex) ← Forward (LM loss)
P(age | codes)      ← Reverse (age prediction loss)
P(sex | codes)      ← Reverse (sex prediction loss)
```

**How it works:**
```python
# During training
age=0, codes=[V3000, 769, V290]  ← Newborn
decoder_hiddens = encode(codes)
predicted_age = age_predictor(decoder_hiddens) = 0.5  ✓ Correct!
age_loss = (0.5 - 0)² = 0.25  ← Low loss

# If model tries to generate adult codes:
age=0, codes=[42979, 41401, ...]  ← Adult cardiac codes
predicted_age = age_predictor(decoder_hiddens) = 65  ✗ Wrong!
age_loss = (65 - 0)² = 4225  ← HUGE loss!
         ↓
Gradients prevent decoder from generating age-inappropriate codes
```

**Documentation:**
- **MULTITASK_LEARNING.md** - Full technical analysis with literature evidence
- **MEDICAL_VALIDITY.md** - Validation metrics and age/sex constraints
- **MULTITASK_IMPLEMENTATION.md** - Implementation steps and timeline
