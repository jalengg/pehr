# 11: Model Architecture - BART Encoder-Decoder with Demographic Conditioning

**Estimated Time:** 90 minutes
**Prerequisites:** [07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md), [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)
**Next:** [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md)

---

## Learning Objectives

- Understand BART encoder-decoder architecture (6L-6L-768H)
- Learn embedding layer structure (6,992 tokens × 768 dimensions)
- Understand attention mechanisms (self-attention, cross-attention)
- Learn PromptBartModel with dual prompt conditioning (encoder + decoder)
- Understand why BART is better than GPT for EHR reconstruction
- Learn multi-task learning heads (age prediction, sex prediction)

---

## Why BART for EHR Generation?

### BART vs GPT

**GPT (Decoder-only):**
```
Input: <BOS> 401.9 250.00
       ↓ (causal self-attention)
Output: 250.00 585.9 <EOS>  (next token prediction)
```

**Pros:** Fast autoregressive generation
**Cons:** No bidirectional context, harder to condition on corrupted input

**BART (Encoder-Decoder):**
```
Encoder input:  <BOS> 401.9 <mask> <EOS>  (corrupted, bidirectional)
       ↓
Encoder output: Rich contextual representations
       ↓ (cross-attention)
Decoder output: <BOS> 401.9 250.00 585.9 <EOS>  (reconstruct original)
```

**Pros:**
- Bidirectional encoder (sees full context)
- Denoising objective (robust to missing data)
- Better reconstruction from partial prompts
- Cross-attention enables richer conditioning

**Cons:** Slower generation (two forward passes)

**Why BART for PromptEHR?**
1. **Reconstruction task:** Generate patient from partial code prompts (BART excels at this)
2. **Bidirectional context:** Encoder sees all available codes simultaneously
3. **Denoising training:** Naturally handles missing/corrupted codes
4. **Conditional generation:** Cross-attention integrates demographic prompts effectively

---

## BART Configuration

### Model Size (config.py:15-30)

```python
config = BartConfig(
    vocab_size=6992,           # 7 special + 6985 codes
    d_model=768,               # Hidden dimension
    encoder_layers=6,          # Encoder depth
    decoder_layers=6,          # Decoder depth
    encoder_attention_heads=12,  # Multi-head attention
    decoder_attention_heads=12,
    encoder_ffn_dim=3072,      # FFN intermediate size (4 × d_model)
    decoder_ffn_dim=3072,
    max_position_embeddings=512,  # Max sequence length
    dropout=0.1,
    attention_dropout=0.1,
    activation_function='gelu',
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2
)
```

**Model size: 6L-6L-768H**
- 6 encoder layers
- 6 decoder layers
- 768-dimensional hidden states

**Comparison:**
| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|------------|
| BART-base | 6-6 | 768 | 12 | 140M |
| BART-large | 12-12 | 1024 | 16 | 400M |
| **PromptEHR (ours)** | 6-6 | 768 | 12 | ~65M |

**Why smaller (65M vs 140M)?**
- Smaller vocabulary (6,992 vs 50,265)
- Embeddings: 6,992 × 768 = 5.4M vs 38.6M
- Output projection: Same reduction
- Core transformer layers: Same size

---

## Embedding Layer

### Token Embeddings (prompt_bart_model.py:39)

```python
# Shared embedding table (used by encoder and decoder)
self.model.shared = nn.Embedding(
    num_embeddings=6992,  # Total vocabulary size
    embedding_dim=768,    # d_model
    padding_idx=0         # <PAD> token
)
```

**Mapping:**
```
Token ID → 768-dimensional vector

0 (<PAD>)   → [0.12, -0.45, ..., 0.89]  (768 dims)
1 (<BOS>)   → [0.34, 0.21, ..., -0.12]
7 (401.9)   → [-0.55, 0.78, ..., 0.34]
8 (250.00)  → [0.91, -0.22, ..., 0.67]
...
```

**Why shared?**
- Encoder and decoder use same embedding table
- Reduces parameters (only one embedding table, not two)
- Ensures consistent code representations across encoder/decoder

### Positional Embeddings

**BART uses learned positional embeddings (not sinusoidal):**

```python
position_embeddings = nn.Embedding(
    num_embeddings=512,  # Max sequence length
    embedding_dim=768
)
```

**Final embedding:**
```
final_embedding = token_embedding + position_embedding

Example for token ID=7 at position=2:
  token_emb[7] + pos_emb[2] → [768-dim vector]
```

**Why positional embeddings matter:**
- Transformers have no inherent sequence order (unlike RNNs)
- Position embeddings inject temporal information
- Visit order matters (disease progression)

---

## PromptBartModel Structure

### Architecture (prompt_bart_model.py:16-69)

```python
class PromptBartModel(BartForConditionalGeneration):
    """BART model with demographic prompt conditioning for EHR generation."""

    def __init__(
        self,
        config: BartConfig,
        n_num_features: Optional[int] = None,      # 1 (age)
        cat_cardinalities: Optional[list[int]] = None,  # [2] (gender: M/F)
        d_hidden: int = 128,                       # Reparameterization dim
        prompt_length: int = 1                     # Number of prompt vectors
    ):
        super().__init__(config)

        # Replace encoder and decoder with prompt-aware versions
        self.model.encoder = PromptBartEncoder(config, self.model.shared)
        self.model.decoder = PromptBartDecoder(config, self.model.shared)

        # DUAL prompt encoders (separate for encoder and decoder)
        self.encoder_prompt_encoder = ConditionalPrompt(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            hidden_dim=config.d_model,  # 768
            d_hidden=d_hidden,          # 128
            prompt_length=prompt_length  # 1
        )
        self.decoder_prompt_encoder = ConditionalPrompt(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            hidden_dim=config.d_model,
            d_hidden=d_hidden,
            prompt_length=prompt_length
        )
```

**Key components:**
1. **Token embeddings:** `self.model.shared` (6,992 × 768)
2. **PromptBartEncoder:** 6 transformer layers with self-attention
3. **PromptBartDecoder:** 6 transformer layers with self-attention + cross-attention
4. **Encoder prompt encoder:** Converts demographics → encoder prompt vectors
5. **Decoder prompt encoder:** Converts demographics → decoder prompt vectors (separate!)

### Why Dual Prompt Encoders?

**Alternative 1 (NOT USED): Shared prompt encoder**
```python
# Single prompt encoder for both encoder and decoder
prompt_encoder = ConditionalPrompt(...)
encoder_prompts = prompt_encoder(x_num, x_cat)
decoder_prompts = encoder_prompts  # Reuse same prompts
```

**Problem:** Same demographic representation for encoder and decoder
- Encoder needs: Context-gathering prompts (what codes are present?)
- Decoder needs: Generation-guiding prompts (what codes to generate?)
- Different roles require different parameters

**Our approach (USED): Separate prompt encoders**
```python
encoder_prompts = encoder_prompt_encoder(x_num, x_cat)
decoder_prompts = decoder_prompt_encoder(x_num, x_cat)
```

**Benefits:**
- Encoder prompts learn to attend to diagnostically relevant codes
- Decoder prompts learn to guide generation of age/sex-appropriate codes
- More parameters (2× prompt encoders) but better conditioning

---

## Encoder Architecture

### PromptBartEncoder (prompt_bart_encoder.py)

```python
class PromptBartEncoder(BartEncoder):
    """BART encoder with demographic prompt prepending."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,  # Demographic prompts
        ...
    ) -> Tuple[torch.FloatTensor]:
```

**Processing steps:**

1. **Token embedding:**
   ```python
   # input_ids: [batch, seq_len]
   inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len, 768]
   ```

2. **Prepend demographic prompts:**
   ```python
   if prompt_embeds is not None:
       # prompt_embeds: [batch, num_prompts, 768]  (num_prompts=1)
       inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
       # New shape: [batch, num_prompts + seq_len, 768]
   ```

3. **Add positional embeddings:**
   ```python
   positions = self.embed_positions(input_ids)
   hidden_states = inputs_embeds + positions
   ```

4. **Apply 6 transformer layers:**
   ```python
   for encoder_layer in self.layers:  # 6 layers
       hidden_states = encoder_layer(
           hidden_states,
           attention_mask=attention_mask  # Self-attention
       )
   ```

5. **Return final hidden states:**
   ```python
   return hidden_states  # [batch, num_prompts + seq_len, 768]
   ```

**Encoder output:**
- Shape: `[batch_size, num_prompts + seq_len, 768]`
- Contains: Rich bidirectional representations of input codes
- Used by: Decoder cross-attention

---

## Decoder Architecture

### PromptBartDecoder (prompt_bart_decoder.py)

```python
class PromptBartDecoder(BartDecoder):
    """BART decoder with demographic prompt prepending and cross-attention."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # From encoder
        prompt_embeds: Optional[torch.FloatTensor] = None,
        ...
    ) -> Tuple[torch.FloatTensor]:
```

**Processing steps:**

1. **Token embedding:**
   ```python
   inputs_embeds = self.embed_tokens(input_ids)  # [batch, tgt_len, 768]
   ```

2. **Prepend demographic prompts:**
   ```python
   if prompt_embeds is not None:
       inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
       # [batch, num_prompts + tgt_len, 768]
   ```

3. **Add positional embeddings:**
   ```python
   positions = self.embed_positions(input_ids)
   hidden_states = inputs_embeds + positions
   ```

4. **Apply 6 transformer layers with self-attention and cross-attention:**
   ```python
   for decoder_layer in self.layers:  # 6 layers
       # Self-attention (causal)
       hidden_states = decoder_layer.self_attn(hidden_states, causal_mask=True)

       # Cross-attention (attend to encoder output)
       hidden_states = decoder_layer.encoder_attn(
           hidden_states,
           encoder_hidden_states=encoder_hidden_states  # From encoder
       )

       # Feed-forward
       hidden_states = decoder_layer.fc(hidden_states)
   ```

5. **Return final hidden states:**
   ```python
   return hidden_states  # [batch, num_prompts + tgt_len, 768]
   ```

### Cross-Attention Mechanism

**Self-attention (decoder only):**
```python
# Each decoder token attends to previous decoder tokens
Q, K, V = hidden_states  # All from decoder
attention = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Cross-attention (decoder attends to encoder):**
```python
# Decoder queries, encoder keys/values
Q = hidden_states_decoder        # From decoder
K, V = encoder_hidden_states     # From encoder
attention = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Why cross-attention matters:**
- Decoder can "look at" encoder's bidirectional representations
- Enables reconstruction: Decoder sees corrupted input through encoder
- Enables conditioning: Encoder prompts influence decoder generation

---

## Multi-Task Learning Heads

### Language Model Head (Standard BART)

```python
# Project decoder output to vocabulary logits
lm_head = nn.Linear(768, 6992)  # d_model → vocab_size

# Decoder output: [batch, seq_len, 768]
logits = lm_head(decoder_output)  # [batch, seq_len, 6992]

# Softmax over vocabulary (next token prediction)
probs = softmax(logits, dim=-1)  # [batch, seq_len, 6992]
```

**Loss:**
```python
lm_loss = CrossEntropyLoss(logits.view(-1, 6992), labels.view(-1))
```

### Age Prediction Head (Auxiliary Task)

```python
# Project decoder output to age prediction
age_head = nn.Linear(768, 1)  # d_model → 1 (regression)

# Use <EOS> token representation
eos_hidden = decoder_output[:, -1, :]  # [batch, 768]
age_pred = age_head(eos_hidden)        # [batch, 1]

# MSE loss
age_loss = MSELoss(age_pred, true_age)
```

### Sex Prediction Head (Auxiliary Task)

```python
# Project decoder output to sex classification
sex_head = nn.Linear(768, 2)  # d_model → 2 classes (M/F)

# Use <EOS> token representation
eos_hidden = decoder_output[:, -1, :]  # [batch, 768]
sex_logits = sex_head(eos_hidden)      # [batch, 2]

# Cross-entropy loss
sex_loss = CrossEntropyLoss(sex_logits, true_sex)
```

**Why multi-task learning?**
- Encourages model to encode demographic information
- Improves medical validity (age/sex-appropriate codes)
- Auxiliary losses provide weak supervision
- See [13_MULTI_TASK_LEARNING.md](13_MULTI_TASK_LEARNING.md) for details

---

## Forward Pass Example

### Input

```python
batch = {
    'input_ids': tensor([[1, 3, 7, 6, 4, 2]]),    # <BOS> <v> 401.9 <mask> <\v> <EOS>
    'labels': tensor([[1, 3, 7, 8, 4, 2]]),       # <BOS> <v> 401.9 250.00 <\v> <EOS>
    'x_num': tensor([[65.0]]),                    # Age
    'x_cat': tensor([[0]])                        # Gender (M)
}
```

### Processing

**Step 1: Generate demographic prompts**
```python
encoder_prompt = encoder_prompt_encoder(x_num=65.0, x_cat=0)
# Shape: [1, 1, 768]  (batch=1, num_prompts=1, d_model=768)

decoder_prompt = decoder_prompt_encoder(x_num=65.0, x_cat=0)
# Shape: [1, 1, 768]
```

**Step 2: Encoder forward pass**
```python
# Embed input tokens
token_embeds = embed_tokens([1, 3, 7, 6, 4, 2])  # [1, 6, 768]

# Prepend encoder prompt
encoder_input = cat([encoder_prompt, token_embeds], dim=1)  # [1, 7, 768]

# Apply 6 encoder layers (bidirectional self-attention)
encoder_output = encoder(encoder_input)  # [1, 7, 768]
```

**Step 3: Decoder forward pass**
```python
# Prepare decoder input (shift labels right)
decoder_input_ids = shift_right([1, 3, 7, 8, 4, 2])  # [2, 1, 3, 7, 8, 4]

# Embed decoder tokens
decoder_token_embeds = embed_tokens(decoder_input_ids)  # [1, 6, 768]

# Prepend decoder prompt
decoder_input = cat([decoder_prompt, decoder_token_embeds], dim=1)  # [1, 7, 768]

# Apply 6 decoder layers (causal self-attn + cross-attn to encoder)
decoder_output = decoder(
    decoder_input,
    encoder_hidden_states=encoder_output  # Cross-attention
)  # [1, 7, 768]
```

**Step 4: Compute logits and loss**
```python
# Language model head
logits = lm_head(decoder_output)  # [1, 7, 6992]

# Cross-entropy loss (compare to labels)
lm_loss = CrossEntropyLoss(logits, labels=[1, 3, 7, 8, 4, 2])

# Auxiliary losses (from <EOS> token)
age_pred = age_head(decoder_output[:, -1, :])
sex_pred = sex_head(decoder_output[:, -1, :])
age_loss = MSELoss(age_pred, 65.0)
sex_loss = CrossEntropyLoss(sex_pred, 0)

# Total loss
total_loss = lm_loss + 0.001 * age_loss + 0.001 * sex_loss
```

---

## Model Parameters Breakdown

### Parameter Count

**Embeddings:**
- Token embeddings: 6,992 × 768 = 5.4M
- Position embeddings (encoder): 512 × 768 = 0.4M
- Position embeddings (decoder): 512 × 768 = 0.4M
- **Subtotal: 6.2M**

**Encoder (6 layers):**
- Self-attention: 768 × (4 × 768) × 12 heads × 6 layers = 17M
- FFN: 768 × 3072 × 2 × 6 layers = 28M
- Layer norms: negligible
- **Subtotal: 45M**

**Decoder (6 layers):**
- Self-attention: 17M (same as encoder)
- Cross-attention: 17M
- FFN: 28M
- **Subtotal: 62M**

**Output heads:**
- LM head: 768 × 6,992 = 5.4M
- Age head: 768 × 1 = 768
- Sex head: 768 × 2 = 1,536
- **Subtotal: 5.4M**

**Conditional prompts:**
- Encoder prompt encoder: ~50K
- Decoder prompt encoder: ~50K
- **Subtotal: 0.1M**

**Total: ~119M parameters**

---

## Try It Yourself

### Exercise 1: Inspect Model Structure

```python
from prompt_bart_model import PromptBartModel
from transformers import BartConfig

# Create config
config = BartConfig(
    vocab_size=6992,
    d_model=768,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=12,
    decoder_attention_heads=12
)

# Create model
model = PromptBartModel(
    config,
    n_num_features=1,       # Age
    cat_cardinalities=[2],  # Gender (M/F)
    d_hidden=128,
    prompt_length=1
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Inspect structure
print(model)
```

[IN PROGRESS - Additional exercises on forward pass, embedding lookup, attention visualization]

---

## Summary

**PromptBartModel combines BART encoder-decoder with demographic conditioning:**

1. **BART Architecture:** 6L-6L-768H encoder-decoder with 119M parameters
2. **Embeddings:** 6,992 tokens × 768 dimensions, shared across encoder/decoder
3. **Dual Prompts:** Separate encoder and decoder prompt encoders for stronger conditioning
4. **Encoder:** 6 layers with bidirectional self-attention, prepended demographic prompts
5. **Decoder:** 6 layers with causal self-attention + cross-attention to encoder
6. **Multi-Task Heads:** LM head (vocab prediction) + age head + sex head
7. **Cross-Attention:** Decoder attends to encoder's bidirectional representations

**Key Files:**
- `prompt_bart_model.py:16-140` - PromptBartModel class
- `prompt_bart_encoder.py` - PromptBartEncoder with prompt prepending
- `prompt_bart_decoder.py` - PromptBartDecoder with cross-attention
- `config.py:15-30` - BART configuration

---

## What's Next?

**Next:** [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md) - ConditionalPrompt architecture, reparameterization trick, demographic encoding

**Alternative:**
- [13_MULTI_TASK_LEARNING.md](13_MULTI_TASK_LEARNING.md) - Age/sex prediction heads
- [14_LOSS_FUNCTIONS.md](14_LOSS_FUNCTIONS.md) - LM loss, auxiliary losses, total loss

---

**Navigation:**
- ← Back to [10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md)
- → Next: [12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
