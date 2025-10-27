# PromptEHR vs Our Implementation: Architecture Comparison

**Date:** October 10, 2025
**Purpose:** Understand why PromptEHR succeeds with BART where our implementation fails

---

## Executive Summary

PromptEHR uses a fundamentally different approach from our implementation:

**Our approach:**
- Demographics and codes as text tokens in sequences
- Format: `{age} {race} {sex} <demo> <v> {ICD_codes} <\v> <END>`
- BART treats this as text generation with special tokens

**PromptEHR approach:**
- Demographics encoded as **conditional prompt embeddings** (not text)
- Medical codes have **separate vocabularies** per modality (diag/med/prod)
- Format: Conditional prompts prepended to encoder/decoder + structured code sequences
- BART modified to accept external conditioning

**Key insight:** PromptEHR doesn't mix demographics with medical codes in the token sequence. Demographics are learned embeddings that condition generation, not tokens the model must generate.

---

## 1. Data Format Differences

### Our Implementation

**Sequence format:**
```
"65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 <\v> <END>"
```

**Tokenization result:**
```python
['65', ' WHITE', ' M', ' ', '<demo>', ' ', '<v>', ' 401', '.', '9', ' 250', '.', '00', ' ', '<\v>', ...]
```

**Issues:**
1. ICD codes fragmented by BART tokenizer: `401.9` → `['401', '.', '9']`
2. Demographics mixed with medical data in same sequence
3. Model must learn to distinguish demographics from diagnosis codes
4. BART's pretrained weights optimize for natural language, not medical codes

### PromptEHR Implementation

**Data structure** (`data.py:26-95`):
```python
data = {
    'x': np.ndarray,           # Tabular patient features (age, demographics)
    'v': list or np.ndarray,   # Visit sequences: [[diag_codes], [med_codes], [proc_codes]]
    'y': np.ndarray            # Labels (if prediction task)
}

metadata = {
    'voc': {                   # Separate vocabularies
        'diag': Voc(),         # Diagnosis code vocabulary
        'med': Voc(),          # Medication vocabulary
        'prod': Voc()          # Procedure vocabulary
    },
    'visit': {'mode': 'dense', 'order': ['diag', 'prod', 'med']},
    'max_visit': 20
}
```

**Visit format:**
```python
# Dense format - NOT tokenized as text
visit = [
    [[diag1, diag2, diag3], [med1, med2], [proc1]],  # Visit 1
    [[diag4, diag5], [med3, med4, med5], [proc2]],   # Visit 2
]
```

**Key differences:**
1. Medical codes stored as **integer indices** in separate vocabularies
2. Each code type (diag/med/prod) has its own vocabulary: `Voc.word2idx`, `Voc.idx2word`
3. Demographics stored separately in `x` (numerical/categorical features)
4. No text tokenization of medical codes - direct integer IDs

---

## 2. Model Architecture Differences

### Our Implementation

**Architecture:**
- Vanilla BART encoder-decoder (`facebook/bart-base`)
- Added 4 special tokens: `<demo>`, `<v>`, `<\v>`, `<END>`
- Embeddings resized: 50265 → 50269 tokens
- Initialization: Mean of pretrained embeddings

**Forward pass:**
```python
# Encoder input: demographics + special tokens + medical codes (all as text)
input_ids = tokenizer("65 WHITE M <demo> <v> 401.9 250.00 <\v> <END>")

# Decoder: autoregressive generation with teacher forcing
outputs = model(input_ids=input_ids, labels=labels)
```

**Problem:**
- Encoder receives mixed semantic signal (demographics + codes)
- No explicit conditioning mechanism
- Pretrained BART weights designed for fluent text, not structured data

### PromptEHR Implementation

**Architecture modifications** (`modeling_promptbart.py`):

1. **Custom PromptBartEncoder** (lines 210-217):
```python
class PromptBartEncoder(BartEncoder):
    def forward(self, input_ids, inputs_prompt_embeds=None, ...):
        # Prepend conditional prompt embeddings to input
        if inputs_prompt_embeds is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            # Adjust attention mask for prepended prompts
```

2. **Custom PromptBartDecoder** (lines 338-345):
```python
class PromptBartDecoder(BartDecoder):
    def forward(self, input_ids, inputs_prompt_embeds=None, ...):
        # Prepend conditional prompt embeddings to decoder input
        if inputs_prompt_embeds is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
```

3. **Conditional Prompt Modules** (lines 58-165):
```python
class NumericalConditionalPrompt(nn.Module):
    # Embeds numerical features (age, lab values, etc.)
    def forward(self, x_num):
        return self.embedding(x_num)  # [batch, n_features] → [batch, n_features, hidden_dim]

class CategoricalConditionalPrompt(nn.Module):
    # Embeds categorical features (gender, ethnicity, etc.)
    def forward(self, x_cat):
        return self.embedding(x_cat)  # Uses learnable embeddings per category

class ConditionalPrompt(nn.Module):
    # Combines numerical and categorical prompts
    def forward(self, x_num=None, x_cat=None):
        prompts = []
        if x_num is not None: prompts.append(self.num_prompt(x_num))
        if x_cat is not None: prompts.append(self.cat_prompt(x_cat))
        return torch.cat(prompts, dim=1)  # Concatenate all prompt embeddings
```

**Forward pass:**
```python
# 1. Encode demographics as prompt embeddings (NOT tokens)
prompt_embeds = prompt_encoder(x_num=[age, lab_val], x_cat=[gender, ethnicity])
# Shape: [batch, n_features, hidden_dim]

# 2. Encoder: prompt embeddings + code token embeddings
input_ids = [<diag> diag_1 diag_2 </diag> <med> med_1 </med>]
encoder_outputs = encoder(input_ids, inputs_prompt_embeds=prompt_embeds)

# 3. Decoder: uses encoder outputs + decoder prompt embeddings
decoder_outputs = decoder(decoder_input_ids,
                          encoder_hidden_states=encoder_outputs,
                          inputs_prompt_embeds=prompt_embeds)
```

**Advantages:**
1. Demographics never compete for token space with medical codes
2. Prompt embeddings are **prepended**, not mixed into sequence
3. Attention mechanism can cleanly separate conditioning (prompts) from generation (codes)
4. Each code type gets dedicated vocabulary (no subword fragmentation)

---

## 3. Tokenization Differences

### Our Implementation

**Single BART tokenizer:**
```python
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
tokenizer.add_special_tokens({'additional_special_tokens': ['<demo>', '<v>', '<\v>', '<END>']})
```

**Problems:**
- ICD codes: `V3001` tokenized as `['V', '300', '1']` (3 fragments)
- Decimal codes: `250.00` tokenized as `['250', '.', '00']` (3 fragments)
- Demographics: `WHITE` → `['WHITE']` (single token, but takes vocabulary space)

### PromptEHR Implementation

**Dual tokenization system** (`modeling_config.py`):

1. **DataTokenizer** (BART tokenizer + code vocabularies):
```python
class DataTokenizer(BartTokenizer):
    def __init__(self):
        super().__init__()
        self.code_vocabs = {
            'diag': {},   # diagnosis code → integer ID
            'med': {},    # medication → integer ID
            'prod': {}    # procedure → integer ID
        }
        self.special_token_dict = {
            'diag': ['<diag>', '</diag>'],
            'med': ['<med>', '</med>'],
            'prod': ['<prod>', '</prod>']
        }

    def add_token_to_code_vocab(self, codes, code_type):
        # Add new medical codes to vocabulary
        for code in codes:
            if code not in self.code_vocabs[code_type]:
                new_id = len(self.code_vocabs[code_type])
                self.code_vocabs[code_type][code] = new_id
```

2. **ModelTokenizer** (converts code IDs for model):
```python
class ModelTokenizer:
    def __init__(self, data_tokenizer):
        self.tokenizer_dict = {}
        for code_type in data_tokenizer.code_vocabs:
            # Create separate tokenizer for each code type
            self.tokenizer_dict[code_type] = self._build_tokenizer(
                data_tokenizer.code_vocabs[code_type]
            )
```

**Encoding process:**
```python
# Step 1: Code stored as integer ID in vocabulary
diag_code = 'V3001'
diag_id = data_tokenizer.code_vocabs['diag']['diag_V3001']  # e.g., 1523

# Step 2: Model tokenizer maps to model vocabulary
model_id = model_tokenizer.encode(diag_id, code_type='diag')

# Result: One token per code, no fragmentation
```

**Key insight:** Medical codes are **never passed through BART's text tokenizer**. They maintain separate vocabularies with 1:1 code-to-token mapping.

---

## 4. Generation Process Differences

### Our Implementation

**Generation function** (`main.py:269-364`):
```python
def generate_patient_sequence(model, tokenizer, prompt, temp=1.0, top_k=50, max_len=256):
    # 1. Encode demographic prompt as text
    inputs = tokenizer(prompt, return_tensors='pt')  # "65 WHITE M <demo>"

    # 2. Force first decoder token to be <v>
    initial_decoder_input = torch.tensor([[tokenizer.bos_token_id, v_token_id]])

    # 3. Autoregressive sampling
    output_ids = model.generate(
        inputs.input_ids,
        decoder_input_ids=initial_decoder_input,
        max_length=max_len,
        do_sample=True,
        temperature=temp,
        top_k=top_k,
        eos_token_id=end_token_id,
        bad_words_ids=[[eos_token_id]]  # Block BART EOS
    )

    return tokenizer.decode(output_ids)
```

**Issues:**
1. Demographics as text tokens in encoder input
2. Decoder must learn to transition from `<demo>` → `<v>` → codes
3. No explicit structure enforcement
4. BART's pretrained EOS competes with our special tokens

### PromptEHR Implementation

**Generation function** (`promptehr.py:633-739`):
```python
def _generation_loop(self, data, inputs):
    # 1. Encode demographics as prompt embeddings (once)
    if 'x_num' in data and 'x_cat' in data:
        sample_gen_kwargs['x_num'] = data['x_num']
        sample_gen_kwargs['x_cat'] = data['x_cat']

    # 2. Initialize with random real code
    init_code = random.sample(data['diag'][0], 1)  # Start from real data
    input_ids = [bos, '<diag>', init_code]

    # 3. Generate per visit, per code type
    for visit in range(num_visits):
        for code_type in ['diag', 'med', 'prod']:
            # Get target number of codes from real data
            num_codes = len(data[code_type][visit])

            # Generate with conditional prompts
            new_tokens = self.model.generate(
                input_ids,
                code_type=code_type,  # Tells model which vocabulary to use
                max_length=num_codes + 2,
                **sample_gen_kwargs  # Includes x_num, x_cat
            )

            # Wrap with modality markers
            input_ids = torch.cat([
                input_ids,
                ['<', code_type, '>'],
                new_tokens,
                ['</', code_type, '>']
            ])

        # End visit
        input_ids.append(eos)

    return new_record
```

**Custom sample function** (`generator.py:289-562`):
```python
def sample(self, input_ids, code_type='diag', **kwargs):
    # Key innovation: code_type-specific generation
    while True:
        outputs = self(
            input_ids,
            encoder_outputs=encoder_outputs,
            code_type=code_type  # Model uses code-specific vocabulary
        )

        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature/top-k
        next_token_scores = logits_warper(input_ids, next_token_logits)
        probs = softmax(next_token_scores)
        next_tokens = multinomial(probs)

        # CRITICAL: Convert model ID back to code vocabulary
        new_next_tokens = next_tokens + model_tokenizer.label_offset
        new_next_tokens_str = model_tokenizer.tokenizer_dict[code_type].decode(new_next_tokens)

        # Decode to actual code string (e.g., "1523" → "V3001")
        input_ids = torch.cat([input_ids, torch.tensor([new_next_tokens_str])])

        if next_tokens == eos_token_id:
            break

    return input_ids
```

**Key differences:**
1. Demographics passed as `x_num`/`x_cat` kwargs, not in input sequence
2. Generation happens per code type (separate vocabulary spaces)
3. Structure enforced by generation loop (not learned by model)
4. Model never predicts demographics - only medical codes

---

## 5. Training Differences

### Our Implementation

**Training loop:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Teacher forcing: decoder sees ground truth
        outputs = model(
            input_ids=batch['input_ids'],
            labels=batch['labels']  # Entire sequence including demographics
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Loss calculation:**
```python
# CrossEntropyLoss over all tokens (demographics + codes)
loss = CrossEntropyLoss(logits, labels)
# Model learns to predict: "65" → "WHITE", "WHITE" → "M", "M" → "<demo>",
#                          "<demo>" → "<v>", "<v>" → "401", etc.
```

**Problem:** Model must learn the entire structured format end-to-end, including transitions between semantic domains (demographics → diagnosis codes).

### PromptEHR Implementation

**Training loop** (`trainer.py`):
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Separate inputs for prompts and codes
        x_num = batch['x_num']      # Numerical features
        x_cat = batch['x_cat']      # Categorical features
        input_ids = batch['input_ids']  # Code sequences only

        # Forward: conditional on prompts
        outputs = model(
            input_ids=input_ids,
            x_num=x_num,
            x_cat=x_cat,
            code_type=batch['code_type'],  # Which vocabulary
            labels=batch['labels']
        )

        # Loss: only on code predictions, not demographics
        loss = outputs.loss
```

**Loss per code type** (`evaluator.py`):
```python
# Separate loss calculation for each modality
for code_type in ['diag', 'med', 'prod']:
    logits = outputs.logits
    labels = batch[f'{code_type}_labels']

    # CrossEntropyLoss over code vocabulary for this type
    loss_diag = CrossEntropyLoss(logits, labels, vocab_size=len(vocabs['diag']))
```

**Key differences:**
1. Demographics never appear in labels - model doesn't learn to generate them
2. Separate vocabularies per code type prevent cross-contamination
3. Conditional prompts provide static context (no prediction needed)
4. Loss focuses purely on medical code generation quality

---

## 6. Why Our Approach Fails

### Issue 1: Semantic Overload
**Problem:** BART encoder receives `"65 WHITE M <demo> <v> 401.9 250.00"` as a single sequence.

**Why it fails:**
- BART pretrained on natural language: "The patient is 65 years old..."
- Our format: Non-linguistic token soup mixing numbers, text, and codes
- No clear semantic boundaries for encoder to learn

**Evidence from logs:**
```
Generated: "31 OTHER F .31 OTHER M"
```
Model learned demographics are important but not their role as conditioning variables.

### Issue 2: Token Competition
**Problem:** Our custom tokens (`<demo>`, `<v>`, `<END>`) compete with 50,265 pretrained tokens.

**Quantitative failure:**
```
After 5 epochs:
  <v> probability: 0.004 (0.4%)
  BART EOS probability: 0.968 (96.8%)

After 15 epochs with blocking BART EOS:
  <END> probability: 0.90 (90%)
  But no codes generated between <demo> and <END>
```

**Root cause:** Millions of pretrained gradient updates vs. 2,820 training steps. Even with proper initialization, new tokens can't compete.

### Issue 3: Structural Ambiguity
**Problem:** Model must learn complex grammar from data:
- `<demo>` always followed by `<v>`
- `<v>` always followed by codes
- `<\v>` marks visit end, can be followed by another `<v>` or `<END>`

**What happened:**
- Training loss: 0.0007 (model memorized training sequences)
- Generation: Gibberish (model didn't learn underlying structure)
- Classic overfitting on format, not learning generation rules

### Issue 4: Tokenization Fragmentation
**Problem:** ICD codes split into subword tokens.

**Example:**
```python
Input:  "V3001 250.00 401.9"
Tokens: ['V', '300', '1', '250', '.', '00', '401', '.', '9']
```

**Impact:**
- Model must learn: "V" → "300" → "1" is a single code
- No guarantee generated subwords form valid codes
- Decoder can generate: "V" → "305" → "2" (invalid code V3052)

---

## 7. How PromptEHR Solves These Issues

### Solution 1: Separation of Concerns
**Architecture decision:** Demographics as prompts, not tokens.

**Benefits:**
1. Encoder receives clean medical code sequences
2. Demographics provide static conditioning (like classifier guidance)
3. No need to learn demographic→code transitions
4. Model focuses solely on medical code generation

### Solution 2: Dedicated Vocabularies
**Architecture decision:** Separate vocabulary per code type.

**Benefits:**
1. No subword fragmentation: `V3001` = single token ID
2. No vocabulary pollution: BART tokens unchanged
3. Code-specific generation: model knows it's predicting diagnosis vs medication
4. Training signal focused on relevant vocabulary subset

### Solution 3: Explicit Structure
**Architecture decision:** Generation loop enforces structure.

**Implementation:**
```python
# Structure NOT learned by model, but enforced by code:
for visit in visits:
    output += '<v>'
    output += generate(code_type='diag', max_len=num_diag_codes)
    output += '<\v>'

output += '<END>'
```

**Benefits:**
1. Model never generates structural tokens (only diagnosis codes)
2. No risk of invalid sequences (structure enforced programmatically)
3. Simplifies learning objective: predict next diagnosis code, not structure

### Solution 4: Warm Start with Real Data
**Generation strategy:** Initialize with real code from source patient.

```python
# Start generation with actual code from patient's first visit
init_code = random.sample(data['diag'][0], 1)  # e.g., V3001
input_ids = [bos, '<diag>', init_code]
```

**Benefits:**
1. Grounds generation in realistic distribution
2. Avoids "cold start" problem (generating first code from noise)
3. Maintains statistical properties of source data

---

## 8. Critical Architectural Insights

### Insight 1: BART for EHR ≠ BART for Text
**Our mistake:** Treating EHR generation as text generation.

**Reality:** EHR data is **structured sequential data**, not natural language.

**Implication:** Need to modify BART architecture to handle:
- Discrete code vocabularies (not continuous language)
- Conditional generation (demographics → codes, not text → text)
- Sequential structured data (diagnosis codes per visit, not words)

### Insight 2: Pretrained Weights Are a Liability
**Our mistake:** Assuming pretrained BART would accelerate learning.

**Reality:** BART's pretrained weights encode:
- Language syntax and semantics
- Subword tokenization strategies
- EOS token behavior (end of sentence)

**Conflict:** Medical codes have NO linguistic structure. `401.9` (hypertension) has no semantic relationship to text "four oh one point nine".

**PromptEHR solution:** Keep pretrained encoder/decoder *weights*, but:
1. Replace input embedding layer with code embeddings
2. Add conditional prompt mechanism
3. Override generation logic

### Insight 3: Loss ≠ Generation Quality
**Our experience:**
```
Epoch 15: Avg Loss = 0.0007 (excellent)
Generation: "31 OTHER F .31 OTHER M" (garbage)
```

**Why:** Teacher forcing loss measures next-token prediction accuracy, not:
- Autoregressive stability
- Structural validity
- Semantic coherence

**PromptEHR approach:** Use perplexity metrics per code type + generation quality evaluation.

---

## 9. Recommended Architectural Changes

### Option 1: Adopt PromptEHR Architecture (Recommended)

**Changes needed:**
1. **Separate demographics from sequence:**
   ```python
   # Current
   input_sequence = f"{age} {race} {sex} <demo> <v> {codes} <\v> <END>"

   # New
   x_num = torch.tensor([age])
   x_cat = torch.tensor([race_id, sex_id])
   input_sequence = f"<v> {codes} <\v> <v> {codes} <\v> <END>"
   ```

2. **Add conditional prompt encoder:**
   ```python
   class ConditionalPrompt(nn.Module):
       def __init__(self, n_num_features, cat_cardinalities, hidden_dim):
           self.num_embedding = nn.Linear(n_num_features, hidden_dim)
           self.cat_embeddings = nn.ModuleList([
               nn.Embedding(card, hidden_dim) for card in cat_cardinalities
           ])

       def forward(self, x_num, x_cat):
           num_embeds = self.num_embedding(x_num)  # [batch, hidden_dim]
           cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
           return torch.stack([num_embeds] + cat_embeds, dim=1)  # [batch, n_features, hidden_dim]
   ```

3. **Modify BART encoder to accept prompts:**
   ```python
   class PromptBartEncoder(BartEncoder):
       def forward(self, input_ids, prompt_embeds=None, attention_mask=None):
           inputs_embeds = self.embed_tokens(input_ids)

           if prompt_embeds is not None:
               # Prepend prompts
               inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
               # Extend attention mask
               batch_size, n_prompts = prompt_embeds.shape[:2]
               prompt_mask = torch.ones(batch_size, n_prompts, device=input_ids.device)
               attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

           return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
   ```

4. **Create separate vocabularies for ICD codes:**
   ```python
   class CodeVocabulary:
       def __init__(self):
           self.code2idx = {}
           self.idx2code = {}

       def add_code(self, code):
           if code not in self.code2idx:
               idx = len(self.code2idx)
               self.code2idx[code] = idx
               self.idx2code[idx] = code

       def encode(self, codes):
           return [self.code2idx[c] for c in codes]

       def decode(self, indices):
           return [self.idx2code[i] for i in indices]
   ```

5. **Restructure generation loop:**
   ```python
   def generate(self, x_num, x_cat, num_visits, codes_per_visit):
       prompt_embeds = self.prompt_encoder(x_num, x_cat)

       generated_visits = []
       input_ids = torch.tensor([[self.bos_token_id]])

       for visit_idx in range(num_visits):
           # Generate diagnosis codes only
           diag_output = self.model.generate(
               input_ids,
               prompt_embeds=prompt_embeds,
               code_vocabulary=self.diag_vocab,
               max_length=codes_per_visit[visit_idx],
           )

           # Wrap with visit markers
           visit_tokens = torch.cat([
               torch.tensor([[self.v_token_id]]),
               diag_output,
               torch.tensor([[self.end_v_token_id]])
           ], dim=1)

           input_ids = torch.cat([input_ids, visit_tokens], dim=1)
           generated_visits.append(diag_output)

       # Add final END token
       input_ids = torch.cat([input_ids, torch.tensor([[self.end_token_id]])], dim=1)

       return generated_visits
   ```

### Option 2: Decoder-Only Model (Alternative)

**Rationale:** If encoder-decoder boundary causes issues, switch to GPT-style.

**Changes:**
1. Use `facebook/opt-1.3b` or `gpt2-large`
2. Format as single sequence with special tokens for structure
3. Add prefix tuning for demographics
4. Still use separate code vocabularies

**Pros:**
- Simpler architecture (no encoder/decoder alignment)
- Autoregressive generation more natural

**Cons:**
- Loses BART's denoising pretraining benefits
- Requires longer context (full patient history in decoder)

### Option 3: Hybrid Retrieval-Augmented Generation

**Concept:** Combine small generative model with retrieval.

**Architecture:**
1. Encode patient demographics + partial history
2. Retrieve k similar patients from training data
3. Generate codes conditioned on retrieved examples
4. Use contrastive loss to ensure diversity

**Pros:**
- Grounded in real data (reduces hallucination)
- Can handle rare codes better

**Cons:**
- Requires efficient retrieval infrastructure
- More complex training pipeline

---

## 10. Implementation Roadmap

### Phase 1: Data Preparation (1 day)
- [ ] Create diagnosis code vocabulary (single vocabulary, no procedures/medications)
- [ ] Reformat MIMIC-III data:
  - Demographics → `x_num` (age), `x_cat` (gender, ethnicity)
  - Visits → list of diagnosis code lists only
- [ ] Build tokenizer for diagnosis codes
- [ ] Validate: no code fragmentation

### Phase 2: Model Architecture (2 days)
- [ ] Implement `ConditionalPrompt` module (numerical + categorical)
- [ ] Modify `BartEncoder` to accept prompt embeddings
- [ ] Modify `BartDecoder` to accept prompt embeddings
- [ ] Create diagnosis code embedding layer
- [ ] Test forward pass with dummy data

### Phase 3: Training Pipeline (2 days)
- [ ] Update data collator to separate prompts from codes
- [ ] Implement diagnosis code loss calculation
- [ ] Add perplexity metrics for diagnosis codes
- [ ] Validate: loss decreases, perplexity reasonable

### Phase 4: Generation (2 days)
- [ ] Implement structured generation loop (diagnosis only)
- [ ] Add per-visit generation for diagnosis codes
- [ ] Enforce structure: `<v> diag_1 diag_2 ... diag_n <\v> <v> ... <END>`
- [ ] Test: generates valid diagnosis code sequences

### Phase 5: Evaluation (1 day)
- [ ] Generate 1000 synthetic patients
- [ ] Compare distributions (diagnosis code frequency, visit length, codes per visit)
- [ ] Calculate perplexity on held-out data
- [ ] Qualitative review of generated records

**Total estimated time:** 8 days (1.5 weeks)

**Scope:** Diagnosis codes only. Procedures and medications excluded to simplify initial implementation.

---

## 11. Key Takeaways

1. **Don't mix semantics:** Demographics and medical codes serve different roles. Keep them separate architecturally.

2. **Respect the data structure:** EHR data is not natural language. Don't force it through text tokenization.

3. **Pretrained weights need adaptation:** BART's pretrained encoder/decoder are useful, but input/output layers must be redesigned for structured data.

4. **Enforce structure externally:** Don't make the model learn `<demo> <v> codes <\v> <END>` format. Enforce it in the generation loop.

5. **Low training loss ≠ good generation:** Autoregressive generation stability is different from teacher forcing accuracy.

6. **Separate vocabulary for medical codes:** Dedicated diagnosis code vocabulary prevents fragmentation and ensures 1:1 code-to-token mapping.

7. **Conditional prompts > Text conditioning:** Learned embeddings for demographics are more effective than text tokens for conditioning generation.

---

## References

- PromptEHR paper: [Link if available]
- PromptEHR code: `/u/jalenj4/PromptEHR/`
- Our implementation: `/u/jalenj4/pehr_scratch/main.py`
- BART paper: Lewis et al., 2019
- MIMIC-III documentation: https://mimic.mit.edu/

---

**Prepared by:** Claude Code
**Date:** October 10, 2025
**Status:** Ready for implementation discussion
