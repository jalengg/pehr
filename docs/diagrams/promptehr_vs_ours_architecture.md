# Architecture Comparison: PromptEHR vs Our Implementation

**Date:** 2025-10-29
**Purpose:** Visual comparison of architectures, evaluation pipelines, and metrics

---

## Table of Contents

1. [High-Level Architecture Comparison](#high-level-architecture-comparison)
2. [Model Architecture Diagrams](#model-architecture-diagrams)
3. [Training Pipeline Comparison](#training-pipeline-comparison)
4. [Generation Pipeline Comparison](#generation-pipeline-comparison)
5. [Evaluation Framework Comparison](#evaluation-framework-comparison)
6. [Metrics Comparison Table](#metrics-comparison-table)
7. [Novel Contributions Summary](#novel-contributions-summary)

---

## High-Level Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PromptEHR                                   │
└─────────────────────────────────────────────────────────────────────┘

Input Data:
    Multi-code-type EHRs (diagnosis + procedure + medication)
    Demographics (age, sex, race)

Model:
    BART-base (6L encoder, 6L decoder, 768 hidden)
    + Conditional Prompts (demographics → embeddings)
    + Code-type-specific LM heads (diag, proc, med)

Training:
    Objective: Cross-entropy (reconstruction)
    Augmentation: Span masking + deletion + replacement
    Optimization: AdamW, warmup_ratio=0.06

Evaluation:
    Metrics: Perplexity (spatial + temporal)
    Reported: eval_ppl_{code_type}_{spl|tpl}

Generation:
    Strategy: Visit-by-visit, code-type-by-code-type
    Prompting: ~50% real codes (reconstruction task)
    Output: Multi-code-type synthetic EHRs


┌─────────────────────────────────────────────────────────────────────┐
│                    Our Implementation                                │
└─────────────────────────────────────────────────────────────────────┘

Input Data:
    Single-code-type EHRs (diagnosis only)
    Demographics (age, sex)  [race removed for medical validity]

Model:
    BART-base (6L encoder, 6L decoder, 768 hidden)
    + Conditional Prompts (demographics → embeddings)
    + Single LM head (diagnosis codes)
    + Age prediction head (auxiliary)
    + Sex prediction head (auxiliary)

Training:
    Objective: Cross-entropy + age_loss + sex_loss
    Augmentation: Span masking + deletion + replacement + code shuffling
    Optimization: AdamW, warmup_steps=1000

Evaluation:
    Metrics: Perplexity + Jaccard + JS divergence + Co-occurrence
             + Top-100 overlap + KS tests + Medical validity
    Reported: 10 metrics (7 novel)

Generation:
    Strategy: Visit-by-visit OR single-shot
    Prompting: 0% codes (zero-prompt) OR 0-100% (conditional)
    Output: Single-code-type synthetic EHRs
```

---

## Model Architecture Diagrams

### PromptEHR Architecture

```
                    ┌──────────────────────────┐
                    │   Demographics Input     │
                    │  x_num=[age]  x_cat=[sex,race]
                    └──────────────────────────┘
                               │
                               ↓
                    ┌──────────────────────────┐
                    │  ConditionalPrompt       │
                    │  - NumericalPrompt(age)  │
                    │  - CategoricalPrompt(sex,race)│
                    │  Reparameterization:     │
                    │    Linear(d_hidden=128)  │
                    │    → ReLU → Linear(768)  │
                    └──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ↓                             ↓
     ┌──────────────────────┐     ┌──────────────────────┐
     │  PromptBartEncoder   │     │  PromptBartDecoder   │
     │  Prepend prompts to  │     │  Prepend prompts to  │
     │  input embeddings    │     │  decoder embeddings  │
     │  [prompt||input]     │     │  [prompt||decoder]   │
     │                      │     │                      │
     │  6 Encoder Layers    │────>│  6 Decoder Layers    │
     │  (Self-Attention)    │     │  (Self + Cross Attn) │
     └──────────────────────┘     └──────────────────────┘
                                             │
                                             ↓
                              ┌──────────────────────────┐
                              │  Code-Type-Specific      │
                              │  LM Heads (ModuleDict)   │
                              │  - lm_head['diag']       │
                              │  - lm_head['proc']       │
                              │  - lm_head['med']        │
                              └──────────────────────────┘
                                             │
                                             ↓
                              ┌──────────────────────────┐
                              │  Cross-Entropy Loss      │
                              │  Perplexity (median)     │
                              └──────────────────────────┘
```

### Our Implementation Architecture

```
                    ┌──────────────────────────┐
                    │   Demographics Input     │
                    │    x_num=[age]  x_cat=[sex]
                    │    [Race removed]        │
                    └──────────────────────────┘
                               │
                               ↓
                    ┌──────────────────────────┐
                    │  ConditionalPrompt       │
                    │  - NumericalPrompt(age)  │
                    │  - CategoricalPrompt(sex)│
                    │  Reparameterization:     │
                    │    Linear(d_hidden=128)  │
                    │    → ReLU → Linear(768)  │
                    └──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ↓                             ↓
     ┌──────────────────────┐     ┌──────────────────────┐
     │  PromptBartEncoder   │     │  PromptBartDecoder   │
     │  Prepend prompts to  │     │  Prepend prompts to  │
     │  input embeddings    │     │  decoder embeddings  │
     │  [prompt||input]     │     │  [prompt||decoder]   │
     │                      │     │                      │
     │  6 Encoder Layers    │────>│  6 Decoder Layers    │
     │  (Self-Attention)    │     │  (Self + Cross Attn) │
     └──────────────────────┘     └──────────────────────┘
                                             │
                        ┌────────────────────┴─────────────────────┐
                        ↓                    ↓                     ↓
             ┌────────────────┐   ┌──────────────┐   ┌──────────────┐
             │  LM Head       │   │  Age Head    │   │  Sex Head    │
             │  Linear(768    │   │  Linear(768  │   │  Linear(768  │
             │  → vocab_size) │   │  → n_bins)   │   │  → 2)        │
             │  [5,562 codes] │   │  [Classify   │   │  [M/F]       │
             │                │   │   age bin]   │   │              │
             └────────────────┘   └──────────────┘   └──────────────┘
                        │                    │                     │
                        ↓                    ↓                     ↓
             ┌────────────────────────────────────────────────────┐
             │         Multi-Task Loss Computation                │
             │  total_loss = lm_loss                             │
             │             + age_weight * age_loss                │
             │             + sex_weight * sex_loss                │
             │                                                    │
             │  [Balancing medical validity vs semantic coherence]│
             └────────────────────────────────────────────────────┘
```

**Key Differences:**
1. PromptEHR: Multiple LM heads (one per code type)
2. Ours: Single LM head + auxiliary prediction heads
3. PromptEHR: Single objective (LM)
4. Ours: Multi-task objective (LM + age + sex)

---

## Training Pipeline Comparison

### PromptEHR Training Pipeline

```
MIMIC-III Data (.jsonl)
    ↓
MimicDataset
    ↓
MimicDataCollator
    ├─ Randomly select code_type to augment (diag OR proc OR med)
    ├─ Apply span masking to selected code_type
    │  └─ Poisson(λ=3) span lengths
    ├─ Apply deletion (p=0.15)
    ├─ Apply replacement (p=0.15)
    └─ Build next-span prediction task
    ↓
Batch: {input_ids, labels, label_mask, x_num, x_cat, code_type}
    ↓
BartForEHRSimulation.forward()
    ├─ Encode with conditional prompts
    ├─ Decode with conditional prompts
    ├─ Select LM head by code_type
    └─ Compute loss + perplexity
    ↓
Loss: CrossEntropy(logits, labels)
    ↓
Backpropagation
    ↓
Optimizer Step (AdamW)
```

### Our Training Pipeline

```
MIMIC-III Data (CSV)
    ↓
load_mimic_data() → PatientRecord list
    ↓
PromptEHRDataset
    ├─ Shuffle code order within visits (treat as sets)
    ├─ Apply span masking (Poisson λ=3)
    ├─ Apply deletion (p=0.15)
    ├─ Apply replacement (p=0.15)
    └─ Apply next-visit prediction
    ↓
Batch: {input_ids, labels, x_num, x_cat}
    ↓
PromptBartWithDemographicPrediction.forward()
    ├─ Encode with conditional prompts
    ├─ Decode with conditional prompts
    ├─ Compute LM logits (single head)
    ├─ Compute age logits (token-level classification)
    └─ Compute sex logits (token-level classification)
    ↓
Multi-Task Loss:
    total_loss = lm_loss + age_weight*age_loss + sex_weight*sex_loss
    ↓
Backpropagation
    ↓
Optimizer Step (AdamW)
```

**Key Differences:**
1. PromptEHR: Code-type-specific augmentation
2. Ours: Code shuffling + multi-task loss
3. PromptEHR: Single-objective training
4. Ours: Balancing 3 objectives (LM + age + sex)

---

## Generation Pipeline Comparison

### PromptEHR Generation (Reconstruction)

```
Test Patient (Real EHR)
    diag: [[code1, code2, code3], [code4, code5]]
    proc: [[code6]]
    x_num: [65], x_cat: [0, 2]
    ↓
Sample initial code: code1
    ↓
For each visit:
    For each code_type (diag, proc, med):
        ├─ Get real codes: [code1, code2, code3]
        ├─ Random mask ~50%: keep [code1], mask [code2, code3]
        ├─ Encode prompt: <diag> code1 ...
        ├─ model.generate() with prompt
        │  └─ Sample until </diag>
        ├─ Extract generated codes
        └─ Combine: [code1] + [generated_codes]
    Add </s>
    ↓
Synthetic Patient (Reconstructed)
    diag: [[code1, codeX, codeY], [code4, codeZ]]
    proc: [[code6, codeW]]
```

**Task:** Reconstruction (fill in masked codes)
**Prompting:** ~50% real codes provided
**Difficulty:** Easier (partial ground truth)

### Our Generation (Zero-Prompt)

```
Demographics Only
    x_num: [65], x_cat: [0]  # Age, sex
    ↓
Initialize: <s>
    ↓
For each visit (determined by target structure):
    ├─ Encode: <v> <mask> <mask> <mask>
    ├─ model.generate() from demographics ONLY
    │  └─ Sample until <\v> or max_codes
    ├─ Extract generated codes
    └─ Append: <v> codeA codeB codeC <\v>
    Add </s>
    ↓
Synthetic Patient (Fully Generated)
    visits: [[codeA, codeB, codeC], [codeD, codeE]]
```

**Task:** Zero-shot generation (no code prompts)
**Prompting:** 0% codes (demographics only)
**Difficulty:** Harder (cold start)

---

## Evaluation Framework Comparison

### PromptEHR Evaluation

```
┌─────────────────────────────────────────────┐
│         PromptEHR Evaluation Pipeline        │
└─────────────────────────────────────────────┘

Test Data (SequencePatient)
    ↓
For each code_type (diag, proc, med):
    │
    ├─ Spatial Perplexity (SPL)
    │  ├─ Mask all codes of this code_type
    │  ├─ Predict from other code_types + demographics
    │  └─ Compute: median(exp(-log p))
    │
    └─ Temporal Perplexity (TPL)
       ├─ Mask next visit's codes
       ├─ Predict from previous visits + demographics
       └─ Compute: median(exp(-log p))
    ↓
Results:
    {
        'eval_ppl_diag_spl': 2.34,
        'eval_ppl_diag_tpl': 3.12,
        'eval_ppl_proc_spl': 2.89,
        'eval_ppl_proc_tpl': 3.45,
        'eval_ppl_med_spl': 2.56,
        'eval_ppl_med_tpl': 3.22
    }

Total Metrics: 6 (2 types × 3 code types)
```

### Our Evaluation Framework

```
┌─────────────────────────────────────────────┐
│      Our Comprehensive Evaluation            │
└─────────────────────────────────────────────┘

Generated Data (100 patients, zero-prompt)
    ↓
┌─────────────────────────────────────────────┐
│        1. Medical Validity Metrics           │
├─────────────────────────────────────────────┤
│  • Age-appropriate diagnosis rate (%)        │
│  • Sex-appropriate diagnosis rate (%)        │
│  • Duplicate code rate (%)                   │
│  • Per-code validation against rules         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│       2. Semantic Coherence Metrics          │
├─────────────────────────────────────────────┤
│  • JS Divergence (code frequencies)          │
│  • Top-100 Overlap (Jaccard)                 │
│  • Co-occurrence Score (avg pairwise freq)   │
│  • KS Test (visit count distribution)        │
│  • KS Test (codes per visit distribution)    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│        3. Perplexity Metrics (Optional)      │
├─────────────────────────────────────────────┤
│  • Temporal Perplexity (TPL)                 │
│  • Token accuracy                            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│       4. Reconstruction Metrics (Optional)   │
├─────────────────────────────────────────────┤
│  • Jaccard similarity (prompt-aware)         │
│  • Exact match rate                          │
└─────────────────────────────────────────────┘
    ↓
Results:
    {
        'age_appropriate': 0.99,
        'sex_appropriate': 0.96,
        'duplicate_rate': 0.00,
        'js_divergence': 0.61,
        'top_100_overlap': 0.04,
        'cooccurrence_score': 2.80,
        'visit_dist_pvalue': 0.93,
        'codes_dist_pvalue': 0.94,
        'tpl': 1.01,
        'reconstruction_jaccard': 0.40
    }

Total Metrics: 10+ (7 novel)
```

---

## Metrics Comparison Table

### Evaluation Metrics: PromptEHR vs Ours

| Category | Metric | PromptEHR | Our Implementation | Novel? |
|----------|--------|-----------|-------------------|--------|
| **Perplexity** | Spatial Perplexity (SPL) | ✅ | ✅ | ❌ |
| **Perplexity** | Temporal Perplexity (TPL) | ✅ | ✅ | ❌ |
| **Loss** | Cross-Entropy Loss | ✅ | ✅ | ❌ |
| **Medical Validity** | Age-Appropriate Rate | ❌ | ✅ | **✅ Novel** |
| **Medical Validity** | Sex-Appropriate Rate | ❌ | ✅ | **✅ Novel** |
| **Medical Validity** | Duplicate Rate | ❌ | ✅ | **✅ Novel** |
| **Semantic Coherence** | JS Divergence (Code Freq) | ❌ | ✅ | **✅ Novel** |
| **Semantic Coherence** | Co-occurrence Score | ❌ | ✅ | **✅ Novel** |
| **Semantic Coherence** | Top-100 Overlap | ❌ | ✅ | **✅ Novel** |
| **Semantic Coherence** | KS Test (Visit Distribution) | ❌ | ✅ | **✅ Novel** |
| **Semantic Coherence** | KS Test (Codes Distribution) | ❌ | ✅ | **✅ Novel** |
| **Reconstruction** | Jaccard Similarity | ❌ | ✅ | **✅ Novel** |
| **Reconstruction** | Exact Match Rate | ❌ | ❌ | ❌ |

**Summary:**
- **PromptEHR:** 3 metrics (perplexity variants + loss)
- **Our Implementation:** 12 metrics (9 novel contributions)
- **Novel Metrics:** 7 core + 2 optional = **9 novel metrics**

### Metric Interpretation Guide

| Metric | Good Value | Bad Value | Interpretation |
|--------|-----------|-----------|----------------|
| **Spatial Perplexity** | < 3.0 | > 5.0 | Lower = better prediction within visit |
| **Temporal Perplexity** | < 4.0 | > 6.0 | Lower = better next-visit prediction |
| **Age-Appropriate Rate** | > 95% | < 90% | % of codes valid for patient age |
| **Sex-Appropriate Rate** | > 95% | < 90% | % of codes valid for patient sex |
| **Duplicate Rate** | 0% | > 1% | % of duplicate codes within visit |
| **JS Divergence** | < 0.3 | > 0.6 | Code frequency distribution match |
| **Co-occurrence Score** | > 20 | < 10 | Avg training freq of generated pairs |
| **Top-100 Overlap** | > 0.5 | < 0.3 | Jaccard of most common codes |
| **Visit Dist (KS p-value)** | > 0.05 | < 0.05 | Match = distributions similar |
| **Codes Dist (KS p-value)** | > 0.05 | < 0.05 | Match = distributions similar |
| **Reconstruction Jaccard** | > 0.6 | < 0.4 | Overlap of real/synthetic codes |

---

## Novel Contributions Summary

### What PromptEHR Provides

**Architecture:**
- ✅ Conditional prompt injection (demographics → embeddings)
- ✅ BART encoder-decoder with prompts prepended
- ✅ Multi-code-type generation (diag, proc, med)

**Training:**
- ✅ Span masking augmentation (Poisson λ)
- ✅ Token deletion/replacement
- ✅ Next-span prediction task

**Evaluation:**
- ✅ Perplexity (spatial + temporal)

**Generation:**
- ✅ Visit-by-visit generation
- ✅ Partial code prompting (~50%)

### Our Novel Contributions

**Evaluation Framework (Primary Contribution):**
1. **Semantic Coherence Suite:**
   - JS Divergence (code frequency distribution)
   - Co-occurrence Score (pairwise code patterns)
   - Top-100 Overlap (common code matching)
   - Distribution Match (KS tests)

2. **Medical Validity Suite:**
   - Age-appropriate diagnosis validation
   - Sex-appropriate diagnosis validation
   - Duplicate detection

3. **Reconstruction Metrics:**
   - Prompt-aware Jaccard similarity
   - Visit-level code overlap

**Training Enhancements:**
1. **Multi-task Learning:**
   - Age prediction auxiliary loss
   - Sex prediction auxiliary loss
   - Balancing medical validity vs semantic coherence

2. **Data Augmentation:**
   - Code shuffling (treat as unordered sets)
   - Prevents positional bias

**Generation Modes:**
1. **Zero-prompt Generation:**
   - Generate from demographics only (no code prompts)
   - Harder task than PromptEHR reconstruction

2. **Flexible Prompting:**
   - Conditional with 0-100% code prompting
   - Supports both reconstruction and generation

---

## Visual Summary: Key Differences

```
┌─────────────────────────────────────────────────────────────┐
│              ARCHITECTURE COMPARISON                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PromptEHR:                                                  │
│  ┌──────┐   ┌──────┐   ┌────────────┐                      │
│  │ Diag │───│ Proc │───│ Med        │  (3 LM heads)        │
│  └──────┘   └──────┘   └────────────┘                      │
│      ↑          ↑            ↑                               │
│  BART Encoder-Decoder + Prompts                             │
│                                                              │
│  Our Implementation:                                         │
│  ┌──────┐   ┌──────┐   ┌──────┐                            │
│  │ Diag │   │ Age  │   │ Sex  │  (1 LM + 2 auxiliary heads)│
│  └──────┘   └──────┘   └──────┘                            │
│      ↑          ↑          ↑                                 │
│  BART Encoder-Decoder + Prompts                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              EVALUATION COMPARISON                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PromptEHR:                                                  │
│  ┌─────────────────┐                                        │
│  │  Perplexity     │  (Only metric)                         │
│  │  - Spatial      │                                        │
│  │  - Temporal     │                                        │
│  └─────────────────┘                                        │
│                                                              │
│  Our Implementation:                                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Perplexity    │  │ Semantic     │  │ Medical      │    │
│  │ - TPL         │  │ Coherence    │  │ Validity     │    │
│  │ - Token Acc   │  │ - JS Div     │  │ - Age Rate   │    │
│  └───────────────┘  │ - Co-occur   │  │ - Sex Rate   │    │
│                     │ - Top-100    │  │ - Duplicates │    │
│                     │ - KS Tests   │  └──────────────┘    │
│                     └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              GENERATION COMPARISON                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PromptEHR (Reconstruction):                                 │
│  Real Patient → Mask 50% codes → Generate → Synthetic       │
│  [code1, code2, code3] → [code1, <mask>, <mask>]            │
│                       → [code1, codeX, codeY]                │
│                                                              │
│  Our Implementation (Zero-Prompt):                           │
│  Demographics → Generate from scratch → Synthetic            │
│  [age=65, sex=M] → <s> <v> ... <\v> ...                    │
│                  → [[codeA, codeB], [codeC]]                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**PromptEHR focuses on:**
- Perplexity-based quality assessment
- Multi-code-type generation
- Reconstruction task (partial prompting)

**Our implementation extends with:**
- **Comprehensive evaluation** (9 novel metrics)
- **Medical validity checks** (age/sex appropriateness)
- **Semantic coherence analysis** (distribution matching, co-occurrence)
- **Multi-task learning** (balancing objectives)
- **Zero-prompt generation** (harder task)

**Key Insight:** PromptEHR uses perplexity as a **proxy** for generation quality. We provide **direct measurements** of semantic coherence and medical validity, offering a more comprehensive quality assessment framework.

**Citation Recommendation:**
> "While PromptEHR (Wang et al., 2023) evaluates generation quality via perplexity, we extend the framework with semantic coherence metrics (JS divergence, co-occurrence analysis, distribution matching) and medical validity checks (age/sex appropriateness). To our knowledge, this is the first comprehensive evaluation of statistical fidelity and clinical plausibility for EHR generation models."

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Related Documents:**
- `docs/reference/original_promptehr_implementation.md` - Complete PromptEHR documentation
- `docs/analysis/semantic_coherence_research.md` - Solutions for semantic coherence

