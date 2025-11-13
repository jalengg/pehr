# Semantic Coherence Evaluation Implementation - Session Summary

## 1. Primary Request and Intent

**Initial Request:** User asked me to "summarize the advantages and disadvantages of our model and the quality of our synthetic data"

**Key User Corrections Throughout Session:**
1. **Jaccard Misinterpretation:** User corrected my interpretation that low Jaccard (0.19) meant quality "collapse":
   - User: "A point about Jaccard score, the goal isn't to predict the exact same basket of codes, but to produce semantically probable codes. If the model is prompted that a patient has visited with heart problems, we expect to see other codes with similar issues, or if there is an injury, we expect codes related to wounds, etc."
   - User: "Jaccard score only measure if we are literally predicting the same basket of codes, but there are so many codes."

2. **Code-per-visit Issue:** User questioned my claim about generation problems:
   - User: "I dont think the code-per-visit issue actually exists, where are you seeing 200 codes being generated?"
   - I verified: `reconstruction_results.txt` showed normal 3-14 codes/visit, issue was specific to `generate_patient_from_demographics()`

3. **Generation Function Redundancy:** User pointed out architectural redundancy:
   - User: "For generate_patient_from_demographics() with per-visit loop, isn't it very similar to a version of the 50% infill task, just adjusted to 100% masking? I dont get why there is a separate method for it."
   - User's solution: "For now, we can sample a patient from the real data, take their age/sex features, preserve the visit structure with complete masking, and then use that to generate_patient_sequence_conditional()."

**Final Implementation Request:** Implement semantic coherence metrics (not Jaccard) to properly evaluate generated data quality

## 2. Key Technical Concepts

### Evaluation Metrics
- **Jaccard Similarity**: Inappropriate metric - treats semantically similar codes as completely different
- **Jensen-Shannon Divergence**: Measures similarity of code frequency distributions (0.0-1.0, lower is better)
- **Kolmogorov-Smirnov Test**: Tests if two distributions are from same underlying distribution (p-value interpretation)
- **Code Co-occurrence**: Measures if code pairs in generated data also appear together in training (indicates clinical plausibility)
- **Top-K Overlap**: Jaccard similarity of most common K codes (tests if model learned frequency structure)

### Generation Modes
- **Conditional reconstruction** (`prompt_prob=0.5`): Use target structure + 50% code prompts
- **Zero-prompt** (`prompt_prob=0.0`): Use target structure, no code prompts, generate all codes
- **From demographics** (REDUNDANT): Should just sample structure from real patient and use `prompt_prob=0.0`

### Architecture Understanding
- Model is **demographic-conditional** (age/sex injected via prompt embeddings)
- NOT truly unconditional (cannot generate demographics from scratch)
- BART encoder largely unused (this is causal LM task, not seq2seq)
- Token-level age prediction works (99% age-appropriate)

## 3. Files and Code Sections

### metrics.py (4 new functions added)

**Why important:** Implements proper semantic coherence evaluation metrics to replace inappropriate Jaccard similarity

**Function 1: `compute_code_frequency_divergence()`**
```python
def compute_code_frequency_divergence(
    generated_patients: list,
    training_patients: list
) -> float:
    """Compute Jensen-Shannon divergence between code frequency distributions.

    Returns:
        JS divergence value (0.0-1.0, lower is better):
            < 0.1: Excellent match
            0.1-0.3: Good match
            > 0.3: Poor match
    """
    from collections import Counter
    from scipy.spatial.distance import jensenshannon

    # Count code frequencies
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_counts = Counter(gen_codes)
    train_counts = Counter(train_codes)

    # Get all unique codes
    all_codes = set(gen_counts.keys()) | set(train_counts.keys())

    # Create probability distributions (with smoothing)
    gen_probs = np.array([gen_counts.get(code, 0) + 1 for code in all_codes])
    train_probs = np.array([train_counts.get(code, 0) + 1 for code in all_codes])

    # Normalize
    gen_probs = gen_probs / gen_probs.sum()
    train_probs = train_probs / train_probs.sum()

    # Compute JS divergence
    js_div = jensenshannon(gen_probs, train_probs)

    return float(js_div)
```

**Function 2: `compute_distribution_match()`**
```python
def compute_distribution_match(
    generated_patients: list,
    training_patients: list
) -> dict:
    """Compare distributional properties using Kolmogorov-Smirnov tests.

    Returns:
        Dictionary with p-values (p > 0.05 = distributions match)
    """
    from scipy.stats import ks_2samp

    # Extract distributions
    gen_visit_counts = [len(p.visits) for p in generated_patients]
    train_visit_counts = [len(p.visits) for p in training_patients]

    gen_codes_per_visit = [len(v) for p in generated_patients for v in p.visits]
    train_codes_per_visit = [len(v) for p in training_patients for v in p.visits]

    # Run KS tests
    visits_result = ks_2samp(gen_visit_counts, train_visit_counts)
    codes_result = ks_2samp(gen_codes_per_visit, train_codes_per_visit)

    return {
        'visits_pvalue': float(visits_result.pvalue),
        'visits_statistic': float(visits_result.statistic),
        'codes_pvalue': float(codes_result.pvalue),
        'codes_statistic': float(codes_result.statistic),
    }
```

**Function 3: `compute_top_k_overlap()`**
```python
def compute_top_k_overlap(
    generated_patients: list,
    training_patients: list,
    k: int = 100
) -> float:
    """Compute Jaccard similarity of top-K most common codes.

    Returns:
        Jaccard similarity (0.0-1.0):
            > 0.7: Excellent overlap
            0.5-0.7: Good overlap
            < 0.5: Poor overlap
    """
    from collections import Counter

    # Count code frequencies
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_counts = Counter(gen_codes)
    train_counts = Counter(train_codes)

    # Get top-K codes
    gen_top_k = set([code for code, _ in gen_counts.most_common(k)])
    train_top_k = set([code for code, _ in train_counts.most_common(k)])

    # Compute Jaccard similarity
    intersection = len(gen_top_k & train_top_k)
    union = len(gen_top_k | train_top_k)

    if union == 0:
        return 0.0

    return intersection / union
```

**Function 4: `compute_cooccurrence_score()`**
```python
def compute_cooccurrence_score(
    generated_patients: list,
    training_patients: list,
    logger: Optional[logging.Logger] = None
) -> float:
    """Compute average co-occurrence frequency of generated code pairs.

    Returns:
        Average co-occurrence count (higher = more plausible):
            > 50: Excellent (codes frequently co-occur)
            20-50: Good (codes sometimes co-occur)
            < 20: Poor (codes rarely co-occur together)
    """
    from collections import defaultdict
    from itertools import combinations

    # Build co-occurrence matrix from training data
    cooccur_counts = defaultdict(int)
    for patient in training_patients:
        for visit in patient.visits:
            if len(visit) < 2:
                continue
            # Count all pairs in visit
            for code_a, code_b in combinations(sorted(visit), 2):
                pair = (code_a, code_b)
                cooccur_counts[pair] += 1

    # Score generated patients
    scores = []
    for patient in generated_patients:
        for visit in patient.visits:
            if len(visit) < 2:
                continue
            # Get all pairs in generated visit
            pairs = list(combinations(sorted(visit), 2))
            # Look up co-occurrence frequency in training
            pair_scores = [cooccur_counts.get(pair, 0) for pair in pairs]
            if pair_scores:
                scores.append(np.mean(pair_scores))

    if len(scores) == 0:
        return 0.0

    avg_score = np.mean(scores)
    return float(avg_score)
```

### evaluate_semantic_coherence.py (NEW FILE)

**Why important:** Main evaluation script that generates 100 patients with zero-prompt and computes all semantic coherence metrics

**Key sections:**
```python
# Generate patients (sample structure from test set, mask all codes)
for i in range(num_to_generate):
    # Sample a random test patient for structure
    target_patient = test_patients[i]

    # Generate with complete masking (prompt_prob=0.0)
    result = generate_patient_sequence_conditional(
        model=model,
        tokenizer=tokenizer,
        target_patient=target_patient,  # Use their structure
        device=device,
        temperature=0.3,
        top_k=40,
        top_p=0.9,
        prompt_prob=0.0,  # Complete masking - no code prompts
        max_codes_per_visit=20
    )

# Compute all 4 metrics
js_div = compute_code_frequency_divergence(generated_patients, train_patients)
dist_match = compute_distribution_match(generated_patients, train_patients)
top_k_overlap = compute_top_k_overlap(generated_patients, train_patients, k=100)
cooccur_score = compute_cooccurrence_score(generated_patients, train_patients, logger)
```

### generate.py (modifications)

**Changes made:**
1. Renamed `generate_patient_unconditional()` → `generate_patient_from_demographics()`
2. Updated docstring to clarify it's NOT truly unconditional (demographic-conditional)
3. Changed default `prompt_prob=0.5` → `prompt_prob=0.0` in `generate_patient_sequence_conditional()`

**Why important:** Clarified that there's no truly unconditional generation (demographics always condition), and simplified by acknowledging `generate_patient_from_demographics()` is redundant with `prompt_prob=0.0`

### config.py

**Previous changes (from earlier in session):**
```python
# Added flag for reconstruction Jaccard during training validation
compute_reconstruction_jaccard: bool = True  # Compute reconstruction Jaccard (prompt-aware)
```

### trainer.py

**Previous changes (from earlier in session):**
Added reconstruction Jaccard computation to validation loop:
```python
# Compute Reconstruction Jaccard if enabled
if config.training.compute_reconstruction_jaccard:
    from metrics import compute_reconstruction_jaccard
    recon_jaccard = compute_reconstruction_jaccard(
        model=model,
        patient_records=val_patient_records,
        tokenizer=tokenizer,
        device=device,
        logger=logger,
        max_samples=100
    )
    val_metrics['reconstruction_jaccard'] = recon_jaccard
    logger.info(f"Reconstruction Jaccard: {recon_jaccard:.4f}")
```

## 4. Errors and Fixes

### Error 1: Missing scipy dependency
**Error:**
```
ModuleNotFoundError: No module named 'scipy'
```

**Location:** When running `evaluate_semantic_coherence.py`, the `compute_code_frequency_divergence()` function tried to import `jensenshannon` from scipy

**Fix:**
```bash
pip install scipy
```

**Result:** Successfully installed scipy-1.16.2, evaluation script ran to completion

### User Feedback on Interpretations

**Misinterpretation 1: Jaccard as quality metric**
- **My error:** Interpreted low Jaccard (0.19 vs 0.42) as "reconstruction quality collapsed"
- **User correction:** "Jaccard score only measure if we are literally predicting the same basket of codes, but there are so many codes"
- **Learning:** Jaccard is inappropriate - semantic coherence is what matters

**Misinterpretation 2: Code-per-visit "problem"**
- **My error:** Claimed there was a major issue with 180-234 codes/visit across all generation
- **User correction:** "I dont think the code-per-visit issue actually exists, where are you seeing 200 codes being generated?"
- **Reality:** Only `test_unconditional.py` had this issue; `reconstruction_results.txt` showed normal 3-14 codes/visit
- **Learning:** Need to verify claims with actual data before stating problems

**Misinterpretation 3: Need for separate generation function**
- **My error:** Proposed complex per-visit loop for `generate_patient_from_demographics()`
- **User correction:** "isn't it very similar to a version of the 50% infill task, just adjusted to 100% masking?"
- **Solution:** Just sample structure from real patient and use `prompt_prob=0.0`
- **Learning:** Avoid overcomplicating when simpler solution exists

## 5. Problem Solving

### Solved: Proper Evaluation Metrics for Synthetic EHR Data

**Problem:**
- Jaccard similarity (0.19) suggested poor quality
- But Jaccard measures exact code matches, not semantic coherence
- Need metrics that capture clinical plausibility

**Solution Implemented:**
1. **Code frequency divergence** (JS divergence): Measures if common codes are common, rare codes are rare
2. **Distribution match** (KS tests): Tests if visit counts and codes-per-visit match training
3. **Top-K overlap**: Tests if model learned most frequent codes
4. **Co-occurrence score**: Tests if code pairs appear together in training (clinical plausibility)

**Results from Evaluation:**
```
Code Frequency Divergence: 0.6078 (Poor - distributions differ significantly)
Visit Distribution Match: p=0.9302 (Match - distributions similar)
Codes Distribution Match: p=0.9388 (Match - distributions similar)
Top-100 Overlap: 0.0390 (Poor - missing common codes)
Co-occurrence Score: 1.79 (Poor - codes rarely co-occur together)
```

**Interpretation:**
- ✓ Visit structure matches training (good)
- ✓ Codes per visit matches training (good)
- ✗ Code frequencies don't match (poor)
- ✗ Top codes don't overlap with training (poor)
- ✗ Code pairs rarely co-occur in training (poor)

**Conclusion:** Model generates structurally correct but semantically implausible sequences. The auxiliary losses (age/sex prediction) may have helped medical validity but hurt semantic coherence.

### Ongoing: Understanding Why Semantic Coherence is Poor

**Open Questions:**
- Why is JS divergence so high (0.61)?
- Why is top-100 overlap so low (0.039 = only ~4 codes overlap)?
- Why do generated code pairs rarely co-occur in training (1.79 avg)?

**Hypotheses:**
1. Auxiliary losses dominate gradients, model learns "age-appropriate" over "realistic combinations"
2. Token-level age prediction creates code-level feedback that disrupts natural code distributions
3. Model may be generating rare/unusual codes that are technically valid but uncommon

## 6. All User Messages

1. "Can you take a look at the training so far and do a sanity check on the statistics? Is learning rate too low or is it good enough? How is the loss rate doing?"

2. "Looks like loss has already plateaued to the 0.01->0.05 range, is there a point to continuing to train?"

3. "Do all of those" [referring to: cancel job, evaluate checkpoint, compare to baseline]

4. "Can you also look into why loss plateaus after epoch 30 or so? Also, don't worry about duplicates so much, we can always dedupe in post-processing, I think that's okay. I would like to focus on age-inappropriate codes being generated. How confident do you think age normalization and age_loss_weight adjustments will have an effect?"

5. "Let's keep epochs to just 30 for now"

6. "Let's go ahead and do that, and implement. maybe reduce epochs to 5."

7. "Can you answer: Why is an encoder needed for this? Can't we just do a decoder-only model? What is the input and what is the output? Can you diagram the architecture of the model as I have it so far?"

8. "I want to plan the following changes: I want to generate completely synthetic patients, what happens if we just generate using zero prompt codes, or all of them masked? I am okay with using infill task for training, but for generation, we should start with completely synthetic patients as feasible."

9. "I have some other things to point out. During training, I want you to keep track of jaccard score (but not necessarily have it move the needle). I want you to use a special kind of score that does not count the number of codes already revealed to the model through the prompt."

10. "The plan should include both the jaccard changes and the fully synthetic patient generation."

11. "Just rename the function, and adjust the provided prompts to 0%"

12. "Can you so far summarize the advantages and disadvantages of our model and the quality of our synthetic data?"

13. "I dont think the code-per-visit issue actually exists, where are you seeing 200 codes being generated? How are you going to devise semantic coherence metrics, seems difficult. Same with temporal progression, how do we evaluate that?"

14. "Yes, if you look at reconstruction_results.txt, the patients being generated dont have this issue"

15. "This is almost good, but for generate_patient_from_demographics() with per-visit loop, isn't it very similar to a version of the 50% infill task, just adjusted to 100% masking? I dont get why there is a separate method for it."

16. "For now, we can sample a patient from the real data, take their age/sex features, preserve the visit structure with complete masking, and then use that to generate_patient_sequence_conditional()."

17. "continue" (after session limit)

18. "continue" (again)

19. "what more is left on your todo list? Can you summarize this conversation to the next agent in a file docs/historical/semantic_coherence_chat.md? Be as detailed as you can so all the context can transfer correctly."

## 7. Pending Tasks

**No explicit pending tasks.** The user requested a summary, which has been completed. The semantic coherence evaluation has been implemented and run successfully.

**Implicit next steps** (not explicitly requested):
- Analyze why semantic coherence is poor (JS divergence 0.61, co-occurrence 1.79)
- Consider model improvements to increase semantic coherence
- Potentially need to retrain or adjust auxiliary loss weights

## 8. Current Work

Immediately before this summary request, the semantic coherence evaluation implementation was completed:

**What was accomplished:**
1. Added scipy dependency (`pip install scipy`)
2. Successfully ran `evaluate_semantic_coherence.py`
3. Generated 100 patients using zero-prompt (prompt_prob=0.0, sampled structure)
4. Computed all 4 semantic coherence metrics:
   - Code Frequency Divergence: 0.6078 (Poor)
   - Visit Distribution Match: p=0.9302 (Match)
   - Codes Distribution Match: p=0.9388 (Match)
   - Top-100 Overlap: 0.0390 (Poor)
   - Co-occurrence Score: 1.79 (Poor)

**Results interpretation:**
The evaluation revealed that while the model generates structurally correct sequences (correct visit counts, correct codes per visit), it generates semantically implausible code combinations. The generated codes:
- Have very different frequency distributions from training (JS divergence 0.61)
- Share almost no overlap with top-100 most common codes (3.9% overlap)
- Rarely co-occur together in training data (1.79 average co-occurrence count)

This suggests the auxiliary losses (age/sex prediction) may have disrupted the model's ability to learn realistic code distributions and co-occurrence patterns, even though they improved medical validity (99% age-appropriate).

## 9. Key Insights

### What Works
- Medical validity (99% age-appropriate, 96% sex-appropriate)
- Structural validity (visit counts, codes-per-visit match training)
- Token-level age prediction successfully guides model
- No duplicate codes within visits (suppressed via generation params)

### What Doesn't Work
- Semantic coherence (JS divergence 0.61, target <0.3)
- Code frequency match (top-100 overlap 3.9%, target >50%)
- Clinical plausibility (co-occurrence 1.79, target >20)
- Generated codes don't reflect real-world prevalence

### Critical User Corrections
1. **Jaccard is wrong metric** - Semantic coherence matters, not exact matches
2. **Verify before claiming problems** - Code-per-visit issue didn't actually exist in reconstruction
3. **Simplify, don't overcomplicate** - Use existing functions with different params instead of writing new ones

### Architecture Insights
- BART encoder-decoder may be overkill for this causal LM task
- Demographic conditioning works (via prompt embeddings)
- Auxiliary losses help medical validity but hurt semantic coherence
- Token-level age prediction may create gradient conflicts with LM loss

## 10. Next Steps (Implicit)

To improve semantic coherence:
1. **Reduce auxiliary loss weights** - Try 0.001, 0.0001 to let LM loss dominate
2. **Analyze generated code frequencies** - Which codes are over/under-represented?
3. **Examine training dynamics** - Do auxiliary losses dominate gradients?
4. **Consider decoder-only architecture** - Remove unused encoder
5. **Add co-occurrence loss** - Explicitly penalize implausible code pairs
6. **Curriculum learning** - Start with LM-only, gradually add auxiliary losses

Priority: Balance medical validity (which we have) with semantic coherence (which we lack).
