# Phase 2: Model Architecture - Implementation Summary

## Overview
Phase 2 implements the PromptEHR-inspired BART architecture that conditions the model on demographic information using continuous prompt embeddings rather than text tokens.

**Status**: ✅ COMPLETE - All tests passing

## Key Insight: Prompts as Conditioning Variables
Instead of encoding demographics as text tokens (e.g., "65 WHITE M"), PromptEHR treats them as **external conditioning variables** that get embedded into continuous vectors and **prepended** to both encoder and decoder sequences.

### Why This Matters
```python
# ❌ Original approach (text-based):
input = tokenizer("65 WHITE M <demo> <v> 401.9 250.00 <\v>")
# Problem: Demographics mixed with medical codes, waste tokens

# ✓ PromptEHR approach (embedding-based):
x_num = [[65.0]]          # Age as continuous value
x_cat = [[0, 2]]          # Gender, ethnicity as IDs
prompt_embeds = prompt_encoder(x_num, x_cat)  # → [batch, 3, 768]
# Result: Demographics as 3 conditioning vectors, separate from medical codes
```

## Implementation Architecture

### 1. ConditionalPrompt Module (`conditional_prompt.py`)

Converts demographics to continuous embeddings:

```
Demographics:                    Prompt Embeddings:
-------------                    ------------------
Age: 65.0 (continuous)    →     [1, 1, 768] via Linear layer
Gender: 0 (categorical)   →     [1, 1, 768] via Embedding(2, 768)
Ethnicity: 2 (categorical) →    [1, 1, 768] via Embedding(6, 768)
                                ↓
                         Concatenate: [1, 3, 768]
```

**Classes:**
- `NumericalConditionalPrompt`: Embeds continuous features (age) using Linear layer
- `CategoricalConditionalPrompt`: Embeds categorical features (gender, ethnicity) using Embedding layers
- `ConditionalPrompt`: Combines both types, outputs [batch, n_prompts, hidden_dim]

**Key Details:**
- `prompt_length=1`: Each feature → 1 prompt vector
- Total prompts: 1 (age) + 2 (gender, ethnicity) = 3 vectors
- Hidden dim: 768 for BART-base compatibility

### 2. PromptBartEncoder (`prompt_bart_encoder.py`)

Custom BART encoder that **prepends** demographic prompts to input sequence:

```
Input Sequence:           [batch, seq_len, 768]
Prompt Embeddings:        [batch, 3, 768]
                          ↓ Concatenate
Encoder Input:            [batch, 3 + seq_len, 768]
                          ↓ Self-Attention Layers
Encoder Output:           [batch, 3 + seq_len, 768]
```

**Key Modifications:**
1. Accepts `inputs_prompt_embeds` parameter
2. Concatenates prompts before input embeddings: `torch.cat([prompts, inputs], dim=1)`
3. Extends attention mask to cover prompt positions
4. Positional embeddings applied to full sequence
5. All transformer layers see prompts as part of the sequence

**Inheritance:** Extends `BartEncoder` to leverage pre-trained weights

### 3. PromptBartDecoder (`prompt_bart_decoder.py`)

Custom BART decoder that:
1. **Prepends prompts** to decoder input (only on first generation step)
2. **Cross-attends** to encoder outputs (which include prompt-conditioned representations)

```
Decoder Input:            [batch, tgt_len, 768]
Prompt Embeddings:        [batch, 3, 768]
                          ↓ Concatenate (first step only)
Decoder States:           [batch, 3 + tgt_len, 768]
                          ↓ Self-Attention (causal)
                          ↓ Cross-Attention to Encoder
                          ↓ Feed-Forward
Decoder Output:           [batch, 3 + tgt_len, 768]
```

**Cache Handling (Important for Generation):**
- Modern transformers (4.x) use `Cache` object for KV-cache
- Cache updates in-place during generation
- Prompts only prepended when `past_key_values is None` (first step)
- Subsequent steps use cached prompt representations

**Key Fixes Applied:**
1. Added `embed_scale` initialization for BART's embedding scaling
2. Fixed positional embedding to accept tensor instead of integer
3. Removed manual cache extraction (handled by Cache object)
4. Fixed attention indexing when `output_attentions=True`

### 4. PromptBartModel (`prompt_bart_model.py`)

Complete seq2seq model combining all components:

**Architecture:**
```python
PromptBartModel
├── prompt_encoder: ConditionalPrompt
│   └── Embeds demographics → prompt_embeds
├── model.encoder: PromptBartEncoder
│   └── Prepends prompts to input
├── model.decoder: PromptBartDecoder
│   └── Prepends prompts to decoder input
└── lm_head: Linear
    └── Projects to vocabulary logits
```

**Forward Pass Flow:**
```python
# 1. Encode demographics (only if no cache)
if past_key_values is None:
    prompt_embeds = prompt_encoder(x_num, x_cat)  # [batch, 3, 768]

# 2. Encoder: process input + prompts
encoder_outputs = encoder(
    input_ids,
    inputs_prompt_embeds=prompt_embeds  # Prepended
)  # Output: [batch, 3 + seq_len, 768]

# 3. Extend encoder attention mask for cross-attention
if prompt_embeds is not None:
    encoder_attention_mask = cat([prompt_mask, attention_mask])

# 4. Decoder: generate with prompts + cross-attention
decoder_outputs = decoder(
    decoder_input_ids,
    encoder_hidden_states=encoder_outputs,
    encoder_attention_mask=encoder_attention_mask,  # Includes prompts
    inputs_prompt_embeds=prompt_embeds  # Prepended to decoder
)  # Output: [batch, 3 + tgt_len, 768]

# 5. Language modeling head
lm_logits = lm_head(decoder_outputs)  # [batch, 3 + tgt_len, vocab_size]

# 6. Slice off prompt positions for loss computation
if prompt_embeds is not None:
    lm_logits = lm_logits[:, n_prompts:, :]  # [batch, tgt_len, vocab_size]

# 7. Compute loss

loss = CrossEntropyLoss(lm_logits, labels)
```

**Generation Support:**
- `prepare_inputs_for_generation()`: Passes demographics through all steps
- `_expand_inputs_for_generation()`: Expands demographics for beam search
- Prompts only computed once (first step), cached thereafter

### 5. Integration with Phase 1 Components

**Compatible with:**
- `DiagnosisVocabulary`: Medical codes as discrete tokens
- `DiagnosisCodeTokenizer`: Structural tokens `<v>`, `<\v>`, `<END>`
- `EHRPatientDataset`: Returns `x_num`, `x_cat` alongside token sequences

**Data Flow:**
```python
# Dataset provides:
batch = {
    'input_ids': [batch, seq_len],           # Medical codes
    'labels': [batch, seq_len],              # Target codes
    'x_num': [batch, 1],                     # Age
    'x_cat': [batch, 2],                     # Gender, ethnicity
}

# Model processes:
outputs = model(
    input_ids=batch['input_ids'],
    labels=batch['labels'],
    x_num=batch['x_num'],
    x_cat=batch['x_cat'],
)
# Returns: loss, logits (prompt-conditioned)
```

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `conditional_prompt.py` | 164 | Embeds demographics to prompt vectors |
| `prompt_bart_encoder.py` | 149 | BART encoder with prompt prepending |
| `prompt_bart_decoder.py` | 208 | BART decoder with prompt prepending & caching |
| `prompt_bart_model.py` | 248 | Complete PromptBART model |
| `test_phase2.py` | 329 | Validation tests |

**Total:** ~1,098 lines of implementation + tests

## Critical Implementation Details

### 1. Prompt Prepending Logic
```python
# In encoder/decoder forward():
if inputs_prompt_embeds is not None:
    # Prepend prompts
    inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)

    # Extend attention mask
    prompt_mask = torch.ones(batch_size, n_prompts, ...)
    attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
```

### 2. Loss Computation Fix
```python
# After lm_head, slice off prompt positions
if prompt_embeds is not None:
    # decoder_outputs: [batch, n_prompts + seq_len, hidden]
    # labels: [batch, seq_len]
    # Need to remove first n_prompts positions
    lm_logits = lm_logits[:, n_prompts:, :]  # Now matches labels shape
```

### 3. Cache Handling (Transformers 4.x)
```python
# Modern API: Cache object, not tuples
layer_outputs = decoder_layer(
    hidden_states,
    past_key_values=past_key_values,  # Cache object
    ...
)
# Returns: (hidden_states, optional_attentions)
# Cache updates in-place, not returned

# Return cache object directly
next_cache = past_key_values if use_cache else None
```

### 4. Generation with Prompts
```python
# Only prepend prompts on FIRST step
if past_key_values is None:  # First step
    prompt_embeds = prompt_encoder(x_num, x_cat)
else:  # Subsequent steps
    prompt_embeds = None  # Already in cache
```

## Test Results

All tests passing ✓:

```
=== Testing ConditionalPrompt ===
✓ ConditionalPrompt output shape correct: [4, 3, 768]
✓ get_num_prompts() correct: 3

=== Testing PromptBartModel ===
✓ Forward pass successful
✓ Logits shape correct: [4, 20, 100]
✓ Loss computed: 4.9981

=== Testing Generation ===
✓ Generation successful: 20 tokens generated

=== Testing with DiagnosisCodeTokenizer ===
✓ Forward pass with DiagnosisCodeTokenizer successful
✓ Model compatible with custom vocabulary
```

## Debugging Journey

### Issue 1: Missing `embed_scale`
**Error:** `AttributeError: 'PromptBartEncoder' object has no attribute 'embed_scale'`
**Cause:** BART scales embeddings by `sqrt(d_model)` when `scale_embedding=True`
**Fix:** Added `__init__` to initialize `embed_scale` in encoder/decoder

### Issue 2: Positional Embedding Type
**Error:** `AttributeError: 'int' object has no attribute 'shape'`
**Cause:** Called `embed_positions(seq_len)` instead of `embed_positions(tensor)`
**Fix:** Pass `inputs_embeds` tensor directly

### Issue 3: Encoder Attention Mask Mismatch
**Error:** `RuntimeError: The size of tensor a (23) must match the size of tensor b (20)`
**Cause:** Encoder had 23 tokens (3 prompts + 20 input) but attention mask had 20
**Fix:** Extended `encoder_attention_mask` in `PromptBartModel.forward()` before passing to decoder

### Issue 4: Cache Indexing
**Error:** `IndexError: tuple index out of range`
**Cause:** Tried to extract cache from `layer_outputs[1]`, but modern API uses Cache object
**Fix:** Removed cache extraction, pass `past_key_values` through, return it unchanged

### Issue 5: Loss Shape Mismatch
**Error:** `ValueError: Expected input batch_size (92) to match target batch_size (80)`
**Cause:** Logits included prompt positions, labels didn't
**Fix:** Sliced logits to remove first `n_prompts` positions before loss computation

### Issue 6: Generation Attention Mask Size
**Error:** `RuntimeError: The size of tensor a (8) must match the size of tensor b (4)`
**Cause:** Prompts prepended on every generation step, causing cache size mismatch
**Fix:** Only prepend prompts when `past_key_values is None` (first step only)

## Key Learnings

1. **Pre-trained weights preserved**: Custom encoder/decoder inherit from BART, keeping all learned parameters
2. **Architecture modification ≠ retraining from scratch**: We add prompt logic but use pre-trained transformer layers
3. **Cache is performance, not architecture**: KV-cache speeds up generation but doesn't affect model behavior
4. **Prompts are one-time conditioning**: Prepend once, cache handles the rest during generation
5. **Attention masks must extend**: Every prepended token needs corresponding mask positions

## Codebase Knowledge Map (Updated)

```
pehr_scratch/
├── Phase 1: Data Preparation ✅
│   ├── vocabulary.py              # DiagnosisVocabulary (1:1 code mapping)
│   ├── data_loader.py             # MIMIC-III loading, PatientRecord
│   ├── code_tokenizer.py          # DiagnosisCodeTokenizer
│   ├── dataset.py                 # EHRPatientDataset, EHRDataCollator
│   └── test_phase1.py             # All tests passing ✓
│
├── Phase 2: Model Architecture ✅
│   ├── conditional_prompt.py      # Demographics → prompt embeddings
│   ├── prompt_bart_encoder.py     # BART encoder + prompt prepending
│   ├── prompt_bart_decoder.py     # BART decoder + prompt prepending + caching
│   ├── prompt_bart_model.py       # Complete PromptBART model
│   └── test_phase2.py             # All tests passing ✓
│
├── Documentation
│   ├── phase1_summary.md          # Phase 1 implementation details
│   └── phase2_summary.md          # Phase 2 implementation details (this file)
│
└── Legacy (deprecated)
    ├── main.py                    # Original text-based BART approach
    └── other old files
```

## What's Next: Phase 3 (Training Pipeline)

Upcoming implementation:
1. **Training loop**: Integrate Phase 1 data with Phase 2 model
2. **Data collation**: Handle variable-length sequences with demographics
3. **Loss monitoring**: Track training/validation loss
4. **Checkpointing**: Save model state during training
5. **Metrics**: Evaluate code prediction accuracy

The architecture is now ready for training on MIMIC-III data!
