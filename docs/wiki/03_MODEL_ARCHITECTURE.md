# Model Architecture

**Last Updated:** October 24, 2025

This document describes the neural network architecture for demographic-conditioned synthetic EHR generation.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                 PromptBartWithDemographicPrediction         │
│                                                             │
│  Input: x_num (age), x_cat (sex), input_ids                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ConditionalPrompt (Reparameterization)       │  │
│  │  - Numerical: weight×age + bias → project to 768    │  │
│  │  - Categorical: embedding + bias → project to 768    │  │
│  │  Output: [batch, 1, 768] prompt vectors              │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│           ┌───────────┴───────────┐                         │
│           ↓                       ↓                         │
│  ┌────────────────┐      ┌────────────────┐                │
│  │ Encoder Prompts│      │ Decoder Prompts│                │
│  │   (separate)   │      │   (separate)   │                │
│  └───────┬────────┘      └───────┬────────┘                │
│          ↓                       ↓                         │
│  ┌────────────────┐      ┌────────────────┐                │
│  │ PromptBart     │      │ PromptBart     │                │
│  │ Encoder        │─────→│ Decoder        │                │
│  │ (6 layers)     │      │ (6 layers)     │                │
│  └────────────────┘      └───────┬────────┘                │
│                                  │                         │
│                       ┌──────────┴──────────┐              │
│                       ↓                     ↓              │
│             ┌──────────────┐      ┌──────────────┐         │
│             │  LM Head     │      │  Age Head    │         │
│             │  (logits)    │      │  (6 classes) │         │
│             └──────────────┘      └──────────────┘         │
│                                          ↓                 │
│                                   ┌──────────────┐         │
│                                   │  Sex Head    │         │
│                                   │  (binary)    │         │
│                                   └──────────────┘         │
│                                                             │
│  Output: lm_logits, age_logits, sex_logits                 │
└─────────────────────────────────────────────────────────────┘

Loss = lm_loss + 0.001×age_loss + 0.001×sex_loss
```

## 1. Base Model: BART

**Model:** `facebook/bart-base` from HuggingFace Transformers

**Architecture:**
- Encoder: 6 layers, 768 hidden dim, 12 attention heads
- Decoder: 6 layers, 768 hidden dim, 12 attention heads
- Vocabulary size: Extended to 5,569 (BART 50,265 + 7 special tokens + 5,562 diagnosis codes)
- Parameters: ~140M (base BART) + ~105M (diagnosis embeddings) + ~67k (prompts) = ~245M total

**Why BART?**
- Encoder-decoder architecture supports both:
  - Bidirectional encoding (for context understanding)
  - Autoregressive decoding (for generation)
- Enables infilling tasks (not just left-to-right generation)
- Matches PromptEHR paper methodology

## 2. Demographic Conditioning

### ConditionalPrompt Module (conditional_prompt.py)

**Purpose:** Transform demographics into prompt embeddings

**Architecture:**

#### Reparameterization Strategy

Instead of direct embedding lookup, demographics pass through a bottleneck:

```python
# Numerical features (age)
age_hidden = weight * age + bias  # [batch, d_hidden=128]
age_prompt = Linear(age_hidden)   # [batch, d_hidden] → [batch, d_model=768]

# Categorical features (sex)
sex_emb = Embedding(sex)          # [batch] → [batch, d_hidden=128]
sex_hidden = sex_emb + bias       # [batch, d_hidden]
sex_prompt = Linear(sex_hidden)   # [batch, d_hidden] → [batch, d_model=768]

# Combine
prompt = age_prompt + sex_prompt  # [batch, d_model=768]
prompt = prompt.unsqueeze(1)      # [batch, 1, d_model]
```

**Why Reparameterization (d_hidden=128)?**
- Regularization: Bottleneck prevents overfitting
- Better gradient flow: Learned transformations instead of direct lookup
- Parameter efficiency: Smaller embedding tables

**Parameters:**
- Numerical: `1×128` (weight) + `1×128` (bias) + `128×768` (projection) = 98,560
- Categorical: `2×128` (embeddings for M/F) + `1×128` (bias) + `128×768` (projection) = 98,816
- Total per encoder: ~197k params
- Dual encoders (encoder + decoder): ~394k params

### Dual Prompt Injection

**Separate prompt encoders for encoder and decoder:**

```python
class PromptBartWithDemographicPrediction(nn.Module):
    def __init__(self, ...):
        # Separate prompt modules
        self.encoder_prompt = ConditionalPrompt(...)
        self.decoder_prompt = ConditionalPrompt(...)  # NOT shared

        self.encoder = PromptBartEncoder(prompt_encoder=self.encoder_prompt)
        self.decoder = PromptBartDecoder(prompt_encoder=self.decoder_prompt)
```

**Why Dual Prompts?**
- Encoder prompts: Influence contextual representation of input codes
- Decoder prompts: Reinforce demographics during generation, reduce demographic drift
- Separate parameters allow encoder/decoder to use demographics differently

## 3. PromptBartEncoder (prompt_bart_encoder.py)

**Purpose:** Extend BART encoder with prompt injection

**Forward Pass:**

```python
def forward(self, input_ids, attention_mask, x_num, x_cat):
    # 1. Generate prompts
    prompts = self.prompt_encoder(x_num, x_cat)  # [batch, 1, 768]

    # 2. Get input embeddings
    embeds = self.embed_tokens(input_ids)  # [batch, seq_len, 768]

    # 3. Concatenate prompts to beginning of sequence
    embeds = torch.cat([prompts, embeds], dim=1)  # [batch, 1+seq_len, 768]

    # 4. Extend attention mask
    prompt_mask = torch.ones(batch, 1)  # Prompt always attends
    attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

    # 5. Pass through BART encoder layers
    for layer in self.encoder.layers:
        embeds = layer(embeds, attention_mask=attention_mask)

    return embeds
```

**Key Points:**
- Prompt prepended to input sequence (acts as additional context)
- Attention mask extended to include prompt
- Prompt participates in all self-attention computations

## 4. PromptBartDecoder (prompt_bart_decoder.py)

**Purpose:** Extend BART decoder with prompt injection

**Forward Pass:**

```python
def forward(self, decoder_input_ids, encoder_hidden_states, x_num, x_cat):
    # 1. Generate prompts (using separate decoder_prompt module)
    prompts = self.prompt_encoder(x_num, x_cat)  # [batch, 1, 768]

    # 2. Get decoder input embeddings
    embeds = self.embed_tokens(decoder_input_ids)  # [batch, tgt_len, 768]

    # 3. Concatenate prompts
    embeds = torch.cat([prompts, embeds], dim=1)  # [batch, 1+tgt_len, 768]

    # 4. Extend attention masks
    decoder_mask = create_causal_mask(1 + tgt_len)  # Prompt + decoder inputs

    # 5. Pass through BART decoder layers
    for layer in self.decoder.layers:
        embeds = layer(
            embeds,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_mask
        )

    return embeds
```

**Key Points:**
- Uses separate `decoder_prompt` module (not shared with encoder)
- Prompt prepended to decoder inputs
- Causal masking ensures prompts don't see future tokens
- Cross-attention to encoder hidden states (including encoder prompts)

## 5. Multi-Task Learning Heads

### Language Modeling Head

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size)
# 768 → 5569 (BART tokens + diagnosis codes)

lm_logits = self.lm_head(decoder_outputs[:, 1:, :])  # Skip prompt position
lm_loss = CrossEntropyLoss(lm_logits, labels)
```

**Purpose:** Predict next token in sequence (primary task)

### Age Prediction Head

```python
self.age_predictor = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 6)  # 6 age brackets
)

age_logits = self.age_predictor(decoder_outputs[:, 1:, :])  # Per-token prediction
age_loss = CrossEntropyLoss(age_logits, age_labels)
```

**Purpose:** Predict age category from generated codes (auxiliary task)

**Age Brackets:**
```python
AGE_BRACKETS = [
    (0, 18),    # Class 0: Pediatric
    (18, 30),   # Class 1: Young adult
    (30, 50),   # Class 2: Middle age
    (50, 65),   # Class 3: Senior
    (65, 80),   # Class 4: Elderly
    (80, 90)    # Class 5: Very elderly
]
```

**Why Token-Level?**
- Each generated code should be age-appropriate
- Model learns age associations for individual codes (e.g., "V3001" → pediatric)

### Sex Prediction Head

```python
self.sex_predictor = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2)  # Binary: M/F
)

sex_logits = self.sex_predictor(decoder_outputs[:, 1:, :])  # Per-token prediction
sex_loss = CrossEntropyLoss(sex_logits, sex_labels)
```

**Purpose:** Predict sex from generated codes (auxiliary task)

**Why Token-Level?**
- Each code should be sex-appropriate
- Model learns sex associations (e.g., pregnancy codes → female)

## 6. Loss Function

### Combined Multi-Task Loss

```python
total_loss = lm_loss + age_loss_weight × age_loss + sex_loss_weight × sex_loss

# Current weights (as of 2025-10-24)
age_loss_weight = 0.001
sex_loss_weight = 0.001
```

**Loss Weighting Strategy:**

| Version | Age Weight | Sex Weight | Result |
|---------|------------|------------|--------|
| v1 | 0.3 | 0.2 | High medical validity (99%), poor semantic coherence |
| v2 | 0.01 | 0.2 | High medical validity (96%), poor semantic coherence |
| v3 | 0.001 | 0.001 | **Current:** Balancing validity vs. semantic coherence |

**Rationale for Low Weights:**
- Primary goal: Learn realistic code distributions (LM loss)
- Auxiliary goal: Weak guidance toward age/sex appropriateness
- High weights (0.2-0.3) dominate LM, destroy semantic coherence

### Label Masking

**Critical:** Padding tokens masked in labels

```python
# During data collation
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

# For mask infilling, only compute loss on masked positions
if task_type == "mask_infilling":
    labels[~mask_positions] = -100  # Only learn to predict masked codes
```

**Why -100?**
- PyTorch CrossEntropyLoss ignores targets with value -100
- Prevents model from learning to predict padding
- For infilling, focuses learning on reconstruction of masked spans

## 7. Forward Pass

### Training Forward

```python
def forward(self, input_ids, attention_mask, labels, x_num, x_cat):
    # 1. Encoder
    encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        x_num=x_num,
        x_cat=x_cat
    )  # [batch, 1+seq_len, 768]

    # 2. Decoder (teacher forcing with labels)
    decoder_input_ids = shift_right(labels)  # Standard seq2seq
    decoder_outputs = self.decoder(
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs,
        x_num=x_num,
        x_cat=x_cat
    )  # [batch, 1+tgt_len, 768]

    # 3. Prediction heads
    lm_logits = self.lm_head(decoder_outputs[:, 1:, :])  # Skip prompt
    age_logits = self.age_predictor(decoder_outputs[:, 1:, :])
    sex_logits = self.sex_predictor(decoder_outputs[:, 1:, :])

    # 4. Compute losses
    lm_loss = F.cross_entropy(
        lm_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )

    age_labels = get_age_bracket(x_num)  # Convert age to class
    age_loss = F.cross_entropy(
        age_logits.view(-1, 6),
        age_labels.view(-1).expand_as(lm_logits[..., 0])
    )

    sex_labels = x_cat[:, 0]  # Binary sex
    sex_loss = F.cross_entropy(
        sex_logits.view(-1, 2),
        sex_labels.view(-1).expand_as(lm_logits[..., 0])
    )

    # 5. Combine
    total_loss = lm_loss + 0.001 * age_loss + 0.001 * sex_loss

    return {
        'loss': total_loss,
        'lm_loss': lm_loss.item(),
        'age_loss': age_loss.item(),
        'sex_loss': sex_loss.item(),
        'logits': lm_logits
    }
```

### Generation Forward

```python
def generate(self, x_num, x_cat, max_length=512, temperature=0.7, top_k=40, top_p=0.9):
    # Uses HuggingFace generate() with custom forward pass
    # Prompts automatically injected via custom encoder/decoder
    # See generate.py for details
```

## 8. Model Parameters

**Total Parameters:** ~245M

**Breakdown:**
- BART base model: ~140M
- Diagnosis code embeddings: ~105M (5,562 codes × 768 dim × 2 for encoder+decoder)
- Prompt modules: ~394k (dual encoders)
- Age/sex heads: ~200k

**Memory Usage:**
- Model parameters: ~1GB (FP32)
- Activations during training (batch=8): ~2GB
- Total: ~3GB GPU memory for model

## 9. Key Design Decisions

### Why Multi-Task Learning?

**Decision:** Add age/sex prediction heads with low weights (0.001)

**Alternatives Considered:**
1. Pure LM: Generates medically invalid codes (30% age violations)
2. High weights (0.2-0.3): Destroys semantic coherence (JS divergence 0.6)
3. Post-processing filters: Loses distributional fidelity

**Trade-off:** Weak auxiliary losses provide soft guidance without dominating LM objective

### Why Reparameterization?

**Decision:** Use d_hidden=128 bottleneck for demographic embeddings

**Alternatives Considered:**
1. Direct embedding lookup: Prone to overfitting, poor generalization
2. No bottleneck: 2× parameters, no regularization benefit

**Trade-off:** Slight parameter increase (~394k) for better gradient flow and regularization

### Why Token-Level Prediction?

**Decision:** Predict age/sex for each generated token (not just sequence-level)

**Alternatives Considered:**
1. Sequence-level prediction: Aggregate over full sequence
2. Visit-level prediction: Aggregate per visit

**Trade-off:** Token-level provides fine-grained supervision (code "V3001" → pediatric) but requires more computation

## 10. Common Issues

### Issue 1: Attention Mask Dimension Mismatch

**Problem:** Prompts prepended but attention mask not extended

**Solution:** Extend masks in both encoder and decoder:
```python
prompt_mask = torch.ones(batch, 1, device=device)
attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
```

### Issue 2: Prompt in Loss Calculation

**Problem:** Including prompt position in loss inflates values

**Solution:** Skip prompt position when computing losses:
```python
lm_logits = self.lm_head(decoder_outputs[:, 1:, :])  # Skip position 0
```

### Issue 3: Auxiliary Losses Dominating

**Problem:** High auxiliary weights (0.2-0.3) destroy semantic coherence

**Solution:** Reduce to 0.001, prioritize LM loss

## Next Steps

- **Understand training process:** See [Training](04_TRAINING.md)
- **Learn about generation:** See [Generation](05_GENERATION.md)
- **See how data flows in:** See [Data Pipeline](02_DATA_PIPELINE.md)
