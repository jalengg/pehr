# Conditional Reconstruction Implementation

**Date**: 2025-10-21
**Purpose**: Migrate from de novo synthesis to PromptEHR-style conditional reconstruction

---

## Changes Summary

Switched generation approach from **unconditional synthesis** (demographics only) to **conditional reconstruction** (demographics + partial real codes + known target length).

---

## Files Modified

### 1. `dataset.py` - Code Shuffling (Lines 395-404)

**Change**: Added random shuffling of code order within visits during training.

**Rationale**:
- Treats diagnosis codes as **unordered sets** rather than sequences
- Prevents positional bias (model learning "important codes appear first")
- Matches original PromptEHR approach
- Enables robust prompting during generation (any random subset works equally well)

**Implementation**:
```python
# Shuffle code order within each visit to treat codes as unordered sets
shuffled_visits = []
for visit in visits:
    if len(visit) > 0:
        shuffled_visit = list(np.random.choice(visit, len(visit), replace=False))
    else:
        shuffled_visit = []
    shuffled_visits.append(shuffled_visit)
```

**Impact**: Model no longer learns sequential dependencies (SEQ_NUM ordering), only code co-occurrence.

---

### 2. `generate.py` - Conditional Reconstruction Function

**New Function**: `generate_patient_sequence_conditional()`

**Replaces**: Previous `generate_patient_sequence()` (de novo approach)

**Key Differences**:

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Input** | Demographics only | Demographics + real patient visits |
| **Prompting** | None | ~50% of real codes per visit |
| **Target Length** | Fixed `max_codes_per_visit=20` | Known from real data |
| **Task** | Pure generation | Reconstruction/infilling |
| **Output** | Single sequence string | Dict with generated/target/prompt visits |

**Algorithm** (per visit):
1. Cap visit codes at `max_codes_per_visit` (20) if needed
2. Randomly mask ~50% using binomial(1, 0.5) sampling
3. Encode prompt: `<v> {kept_codes}`
4. Generate autoregressively until `</v>` or max_new_tokens
5. Extract generated codes (exclude special tokens)
6. Combine: `final = set(generated_codes + prompt_codes)`
7. Ensure exactly `num_target_codes` by resampling if needed

**Post-processing**:
- Deduplication via `set()`
- Length enforcement (resample if too few/many codes)

---

## Files Created

### 1. `split_train_test.py` - Data Split Utility

**Purpose**: Create reproducible train/test splits of MIMIC-III data.

**Features**:
- 80/20 train/test split (configurable)
- Random seed for reproducibility (default: 42)
- Saves split datasets as pickle files
- Logs distribution statistics

**Usage**:
```bash
python split_train_test.py
```

**Output**:
- `data_splits/train_patients.pkl`
- `data_splits/test_patients.pkl`
- `data_splits/vocabulary.pkl`

---

### 2. `generate_from_test.py` - Test Set Reconstruction & Evaluation

**Purpose**: Generate synthetic patients via conditional reconstruction and evaluate quality.

**Features**:
- Loads test patients from split
- Generates reconstructions using `generate_patient_sequence_conditional()`
- Calculates evaluation metrics per patient
- Saves detailed results and aggregate statistics

**Evaluation Metrics**:
1. **Jaccard Similarity**: `|intersection| / |union|` for code sets
2. **Exact Match Rate**: % of visits with perfect reconstruction
3. **Code Count Comparison**: Generated vs target average

**Usage**:
```bash
python generate_from_test.py
```

**Output**:
- `reconstruction_results/reconstruction_results.txt` - Per-patient details
- `reconstruction_results/aggregate_stats.txt` - Summary statistics
- `generate_test.log` - Execution log

**Output Format** (per patient):
```
Patient 1 (ID: 12345)
Demographics: {'age': 65, 'gender': 'M', 'ethnicity': 'WHITE'}
Num Visits: 3
Avg Jaccard: 0.842
Exact Match Rate: 0.333

--- Visit 1 ---
Target Codes (8): ['401.9', '250.00', '428.0', '584.9', ...]
Prompt Codes (4): ['401.9', '428.0', ...]
Generated Codes (8): ['401.9', '250.00', '428.0', '585.9', ...]
Jaccard: 0.875
```

---

## Workflow Changes

### Old Workflow (De Novo Synthesis)

1. Train model on full patient sequences
2. Generate: `model(age, gender, ethnicity) → sequence`
3. Evaluate: Visual inspection only

**Problem**:
- No code count guidance → oversaturation
- No prompting → model must invent everything
- No ground truth → hard to evaluate quality

---

### New Workflow (Conditional Reconstruction)

1. **Split data**: `python split_train_test.py`
   - Train: 2400 patients
   - Test: 600 patients

2. **Train model** on train set (unchanged)

3. **Generate reconstructions**: `python generate_from_test.py`
   - For each test patient:
     - Mask ~50% of codes per visit
     - Generate to reconstruct full visits
   - Compare generated vs real

4. **Evaluate**:
   - Jaccard similarity (how many codes match)
   - Exact match rate (perfect reconstruction)
   - Code count distribution

**Benefits**:
- Realistic code counts (known from real data)
- Easier task (fill gaps, not create from scratch)
- Quantitative evaluation (Jaccard, exact match)
- Direct comparison to original PromptEHR

---

## Expected Results

### Baseline Expectations (from original PromptEHR paper)

- **Jaccard Similarity**: 0.60-0.75 (moderate to good overlap)
- **Exact Match Rate**: 0.10-0.30 (10-30% perfect reconstructions)
- **Code Count**: Should exactly match target (by design)

### Why Not 100% Reconstruction?

- Model only sees ~50% of codes as prompts
- Must infer missing codes from:
  - Demographics (age, gender, ethnicity)
  - Prompt codes (what's already shown)
  - Learned co-occurrence patterns
  - Longitudinal context (previous visits)

### Interpretation

- **High Jaccard (>0.8)**: Model learned strong co-occurrence patterns
- **Low Jaccard (<0.5)**: Model struggling to infer missing codes
- **High Exact Match (>0.3)**: Model memorized common visit patterns
- **Low Exact Match (<0.1)**: Model generates novel code combinations

---

## Key Architectural Decisions

### 1. Code Shuffling During Training
**Tradeoff**: Lose SEQ_NUM priority ordering, gain robustness to prompt order.

**Justification**: PromptEHR treats codes as sets, not sequences. Positional information would create bias during prompting (model expecting important codes first).

### 2. Binomial(1, 0.5) Prompting
**Tradeoff**: Variable prompt length vs fixed %.

**Justification**: Matches original PromptEHR. Creates diverse training signal (sometimes 20% kept, sometimes 80%).

### 3. Set-Based Deduplication
**Tradeoff**: Lose generation artifacts (duplicates) vs lose potential medical meaning.

**Justification**: Same code appearing twice in one visit is medically nonsensical. Using `set()` is cleaner than suppression during generation.

### 4. Length Enforcement (Resampling)
**Tradeoff**: Artificial padding/truncation vs unknown stopping.

**Justification**: Ensures output has exactly `num_target_codes` for fair comparison. Alternative (learn stopping) harder and not focus of current work.

---

## Comparison to Original PromptEHR

| Feature | Original PromptEHR | Our Implementation | Match? |
|---------|-------------------|-------------------|--------|
| Code shuffling | ✅ Yes | ✅ Yes | ✅ |
| Binomial prompting | ✅ Yes (p=0.5) | ✅ Yes (p=0.5) | ✅ |
| Target length known | ✅ Yes | ✅ Yes | ✅ |
| Set deduplication | ✅ Yes | ✅ Yes | ✅ |
| Evaluation metrics | Jaccard, F1 | Jaccard, Exact Match | ~✅ |
| Training tasks | 5 tasks | 5 tasks (same) | ✅ |

**Difference**: Original uses multiple code types (diagnosis + procedure + drug), we only use diagnosis codes currently.

---

## Next Steps

### Immediate (Run Evaluation)

1. Split data: `python split_train_test.py`
2. Generate reconstructions: `python generate_from_test.py`
3. Analyze metrics in `reconstruction_results/aggregate_stats.txt`

### Future Enhancements

1. **Multi-code modalities**: Add procedure codes, medications
2. **Dynamic prompting**: Vary `prompt_prob` per visit (harder = lower prob)
3. **Hierarchical evaluation**: Separate metrics for primary vs secondary diagnoses
4. **Longitudinal metrics**: Evaluate visit-to-visit coherence
5. **Code frequency analysis**: Are common codes reconstructed better?

---

## Troubleshooting

### Issue: Low Jaccard Scores (<0.4)

**Possible causes**:
- Model undertrained (increase epochs)
- Temperature too high (try 0.1-0.3)
- Prompt codes not helping (check if model ignores them)

**Debug**: Print prompt_codes vs generated_codes - are they similar to target?

### Issue: Exact Match Rate = 0%

**Expected for small visit sizes** (harder to get all codes right).

**Concerning if**: Even 1-2 code visits have 0% match → model not learning.

### Issue: Generated codes always same as prompts

**Cause**: Model copying prompt without inference.

**Fix**: Check temperature (too low?), check if model sees prompt in decoder input.

---

## Code Quality Notes

- All functions type-hinted
- Docstrings in Google format
- No deprecated APIs
- No nested functions
- Single Responsibility Principle followed
- Separation of Concerns (generation vs evaluation vs I/O)

---

## Summary

Successfully migrated from **unconditional de novo synthesis** to **conditional reconstruction** matching PromptEHR's approach. Key improvements:

1. ✅ Realistic code counts (known from real data)
2. ✅ Quantitative evaluation (Jaccard, exact match)
3. ✅ Easier task (reconstruction vs generation)
4. ✅ Direct comparison to original paper
5. ✅ Code shuffling prevents positional bias

Ready to run experiments and evaluate model quality.
