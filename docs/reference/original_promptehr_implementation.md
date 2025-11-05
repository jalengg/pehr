# Original PromptEHR Implementation Reference

**Source:** `/u/jalenj/PromptEHR`
**Date Analyzed:** 2025-10-29
**Purpose:** Complete documentation of original PromptEHR codebase for comparison with our implementation

---

## Executive Summary

**PromptEHR is a BART-based conditional EHR generation model** that uses demographic features as conditional prompts to guide generation. Key findings:

**What PromptEHR Includes:**
- ✅ Conditional prompt injection (numerical + categorical features)
- ✅ Multi-code-type generation (diagnosis, procedure, medication)
- ✅ Span masking augmentation (Poisson λ=3)
- ✅ Perplexity-based evaluation (spatial + temporal)
- ✅ Visit-by-visit generation with partial code prompts

**What PromptEHR Does NOT Include:**
- ❌ Semantic coherence evaluation (JS divergence, co-occurrence)
- ❌ Medical validity checks (age/sex appropriateness)
- ❌ Code frequency distribution analysis
- ❌ Top-k code overlap metrics
- ❌ Multi-task learning (only LM objective)
- ❌ Hierarchical ICD-9 generation

**Critical Finding:** PromptEHR evaluates **perplexity only**. All semantic coherence metrics in our implementation are **novel contributions**.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Complete Function Inventory](#complete-function-inventory)
3. [Architecture Details](#architecture-details)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Evaluation Framework](#evaluation-framework)
6. [Generation Procedure](#generation-procedure)
7. [Comparison: PromptEHR vs Our Implementation](#comparison-promptehr-vs-our-implementation)
8. [Data Flow Diagrams](#data-flow-diagrams)

---

## Directory Structure

```
/u/jalenj/PromptEHR/
├── LICENSE
├── README.md
├── demo_data/
│   └── synthetic_ehr/data.pkl        # Pre-generated synthetic data
├── example/
│   └── demo_promptehr.ipynb          # Usage demo notebook
├── pehr/                              # Virtual environment
├── promptehr/                         # Main package directory
│   ├── __init__.py                   # Package exports
│   ├── bart_model.py                 # DEPRECATED (no prompts)
│   ├── constants.py                  # Global constants
│   ├── data.py                       # Data containers
│   ├── dataset.py                    # PyTorch datasets
│   ├── demo_data.py                  # Demo utilities
│   ├── evaluator.py                  # Evaluation logic
│   ├── generator.py                  # Generation mixin
│   ├── model.py                      # Main model
│   ├── modeling_bart.py              # DEPRECATED stub
│   ├── modeling_config.py            # Config & tokenizers
│   ├── modeling_promptbart.py        # Conditional prompts
│   ├── promptehr.py                  # User-facing API
│   └── trainer.py                    # HuggingFace trainer
├── pypi_build_commands.txt
├── requirements.txt
└── setup.py
```

**Core Files:** 13 Python files (2 deprecated)
**Total Lines:** ~3,500 lines of code

---

## Complete Function Inventory

### 1. `promptehr/constants.py`

**Purpose:** Global configuration constants

```python
CODE_TYPES = ['tbd']  # Default code types (overridden by user)
SPECIAL_TOKEN_DICT = {'tbd': ['<tbd>', '</tbd>']}
UNKNOWN_TOKEN = '<unk>'
model_max_length = 512
eps = 1e-8  # For numerical stability

# Pre-trained model URL
PRETRAINED_MODEL_URL = "https://huggingface.co/..."

# Config to training args mapping
config_to_train_args = {
    'max_train_batch_size': 'per_device_train_batch_size',
    'max_eval_batch_size': 'per_device_eval_batch_size',
    # ... (20+ mappings)
}
```

---

### 2. `promptehr/data.py`

**Purpose:** Data containers and vocabulary

#### Class: `Voc`
Vocabulary builder for tokenization.

```python
class Voc:
    def __init__(self):
        self.word2count = {}
        self.idx2word = []
        self.size = 0

    def add_sentence(self, sentence: list):
        """Add tokens from sentence to vocabulary."""
        for word in sentence:
            if word not in self.word2count:
                self.word2count[word] = 1
                self.idx2word.append(word)
                self.size += 1
            else:
                self.word2count[word] += 1
```

#### Class: `SequencePatient`
Patient dataset container.

```python
class SequencePatient:
    def __init__(self, data, metadata):
        """
        Args:
            data: Patient records (x, v, y format)
            metadata: Configuration dict
        """
        self.data = data
        self.config = metadata
        self._parse_inputdata(data)
        self._parse_metadata(metadata)

    def __getitem__(self, index):
        """Return patient record at index."""
        return {
            'visits': self.data['y'][index],  # Codes per visit
            'features': self.data['x'][index],  # Num + cat features
            'pid': index
        }

    def _parse_dense_visit_with_order(self, visits):
        """
        Convert visits from multi-hot tensors to code lists.
        Preserves order information from original data.
        """
        # Implementation details...
```

**Key Methods:**
- `_get_voc_size()` - Calculate vocabulary size per code type
- `_dense_visit_to_tensor()` - Convert code lists to multi-hot tensors
- `_read_pickle()` - Load data from pickle file

---

### 3. `promptehr/dataset.py`

**Purpose:** PyTorch dataset and data collation with augmentation

#### Class: `MimicDataset`

```python
class MimicDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        Load MIMIC-III data splits.

        Args:
            data_dir: Path to data directory
            mode: 'train', 'val', or 'test'
        """
        self.data = self._load_data(data_dir, mode)

    def __getitem__(self, index):
        return self.data[index]
```

#### Class: `MimicDataCollator`

**Purpose:** Batch collation with span masking, deletion, replacement augmentation

```python
class MimicDataCollator:
    def __init__(self, tokenizer, code_types, n_num_feature,
                 mlm_prob=0.15, lambda_poisson=3.0, del_prob=0.15,
                 max_train_batch_size=8, drop_feature=False, mode='train'):
        """
        Initialize collator with augmentation parameters.

        Args:
            mlm_prob: Mask probability for span masking
            lambda_poisson: Poisson λ for span length
            del_prob: Token deletion probability
        """

    def __call__(self, samples):
        """Main collation logic - delegates to mode-specific methods."""
        if self.mode == 'train':
            return self.call_train(samples)
        elif self.mode == 'val':
            return self.call_val(samples)
        else:  # test
            return self.call_test(samples)

    def call_train(self, samples):
        """
        Training collation with augmentation:
        1. Randomly select one code_type to augment
        2. Apply span masking (Poisson λ)
        3. Apply deletion (15% prob)
        4. Apply replacement (15% prob)
        5. Build next-span prediction task (temporal)
        """

    def mask_infill(self, spans):
        """
        Span masking with Poisson-sampled span lengths.
        Similar to BART span corruption.
        """
        # Samples span lengths from Poisson(λ=3)
        # Masks consecutive spans
        # Returns input_ids with <mask> tokens

    def del_token(self, spans):
        """Randomly delete tokens with probability del_prob."""

    def rep_token(self, spans, code_type):
        """Randomly replace tokens from vocabulary with probability rep_prob."""
```

**Key Features:**
- **Span masking:** Poisson(λ=3) for span lengths (matches BART pre-training)
- **Deletion:** 15% token deletion probability
- **Replacement:** 15% token replacement probability
- **Next-span prediction:** Temporal task for multi-visit patients

---

### 4. `promptehr/model.py`

**Purpose:** Main model architecture

#### Class: `BartForEHRSimulation`

```python
class BartForEHRSimulation(BartPretrainedModel):
    def __init__(self, config, model_tokenizer, data_tokenizer):
        """
        Initialize model with code-type-specific LM heads.

        Args:
            config: BartConfig
            model_tokenizer: ModelTokenizer (code-specific IDs)
            data_tokenizer: DataTokenizer (raw record tokenization)
        """
        super().__init__(config)
        self.model = PromptBartModel(config)

        # Create separate LM head for each code type
        self.lm_heads = nn.ModuleDict({
            code: nn.Linear(config.d_model, model_tokenizer.get_num_tokens[code])
            for code in model_tokenizer.get_num_tokens.keys()
        })

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, x_num=None, x_cat=None,
                code_type=None, label_mask=None, **kwargs):
        """
        Forward pass with code-type-specific LM head.

        Returns:
            EHRBartOutput with:
                - loss: Cross-entropy loss
                - logits: Predictions from LM head
                - perplexity: Median perplexity (if label_mask provided)
                - ... (standard BART outputs)
        """
        # Encode + decode with conditional prompts
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            x_num=x_num,
            x_cat=x_cat,
            **kwargs
        )

        # Get code-type-specific logits
        lm_head = self.lm_heads[code_type]
        logits = lm_head(outputs.last_hidden_state)

        # Compute loss
        if labels is not None:
            # Convert labels from data tokenizer IDs to model tokenizer IDs
            encoded_labels = self.model_tokenizer.encode_batch(labels, code_type)
            encoded_labels[labels == -100] = -100  # Preserve mask

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, num_classes), encoded_labels.view(-1))

        # Compute perplexity (if label_mask provided)
        perplexity = None
        if label_mask is not None:
            target = encoded_labels[label_mask.bool()]
            mask_logits = logits[label_mask.bool()]
            prob = torch.gather(mask_logits.softmax(1), 1, target.unsqueeze(-1))
            nll = -torch.log(prob + constants.eps)
            perplexity = nll.exp().median()  # Median for stability

        return EHRBartOutput(
            loss=loss,
            logits=logits,
            perplexity=perplexity,
            # ... standard BART outputs
        )
```

**Key Design Choices:**
1. **Code-type-specific LM heads:** Separate vocabulary per modality (diag, proc, med)
2. **Median perplexity:** More robust than mean to outliers
3. **Label masking:** `label_mask` indicates which positions to compute perplexity on

---

### 5. `promptehr/modeling_promptbart.py`

**Purpose:** Conditional prompt injection

#### Class: `ConditionalPrompt`

```python
class ConditionalPrompt(nn.Module):
    def __init__(self, n_num_feature, cat_cardinalities, d_model, d_hidden=128):
        """
        Encode demographics into prompt embeddings.

        Uses reparameterization trick:
            prompt_embed = W2(relu(W1(features)))

        Args:
            n_num_feature: Number of numerical features (e.g., age)
            cat_cardinalities: List of category counts (e.g., [2] for M/F)
            d_model: BART embedding dimension (768)
            d_hidden: Hidden dimension for reparameterization (128)
        """
        self.num_prompt = NumericalConditionalPrompt(n_num_feature, d_model, d_hidden)
        self.cat_prompt = CategoricalConditionalPrompt(cat_cardinalities, d_model, d_hidden)

    def forward(self, x_num, x_cat):
        """
        Encode features to prompt embeddings.

        Returns:
            prompt_embeds: [batch, n_features, d_model]
        """
        num_embeds = self.num_prompt(x_num) if x_num is not None else None
        cat_embeds = self.cat_prompt(x_cat) if x_cat is not None else None

        if num_embeds is not None and cat_embeds is not None:
            return torch.cat([num_embeds, cat_embeds], dim=1)
        elif num_embeds is not None:
            return num_embeds
        else:
            return cat_embeds
```

#### Class: `PromptBartEncoder`

```python
class PromptBartEncoder(BartEncoder):
    def forward(self, input_ids, attention_mask=None, inputs_prompt_embeds=None, **kwargs):
        """
        Encode inputs with conditional prompts prepended.

        Flow:
            1. Embed input_ids → input_embeds [batch, seq_len, d_model]
            2. If prompts provided, prepend: [prompt_embeds || input_embeds]
            3. Add positional embeddings
            4. Pass through encoder layers
        """
        # Embed tokens
        inputs_embeds = self.embed_tokens(input_ids)

        # Prepend conditional prompts
        if inputs_prompt_embeds is not None:
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            # Extend attention mask for prompt positions
            prompt_mask = torch.ones(batch_size, n_prompts)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Standard BART encoding
        # ...
```

**Key Insight:** Conditional prompts are **prepended to input embeddings**, not added or concatenated to hidden states. This allows prompts to influence all subsequent layers through self-attention.

---

### 6. `promptehr/trainer.py`

**Purpose:** Custom HuggingFace trainer for multi-code-type evaluation

#### Class: `PromptEHRTrainer`

```python
class PromptEHRTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        """
        Evaluate on all code types, compute perplexity for each.

        Returns:
            {
                'eval_ppl_diag': 2.34,
                'eval_ppl_proc': 3.12,
                'eval_ppl_med': 2.89,
                ...
            }
        """
        metrics = {}

        for code_type in self.data_collator.code_types:
            # Set evaluation code type
            self.data_collator.set_eval_code_type(code_type)

            # Compute spatial perplexity
            self.data_collator.set_eval_ppl_type('spl')
            dataloader = self.get_eval_dataloader(eval_dataset, code_type)
            output = self.evaluation_loop(dataloader, description='Evaluation')
            metrics[f'eval_ppl_{code_type}_spl'] = output.metrics['eval_perplexity']

            # Compute temporal perplexity
            self.data_collator.set_eval_ppl_type('tpl')
            dataloader = self.get_eval_dataloader(eval_dataset, code_type)
            output = self.evaluation_loop(dataloader, description='Evaluation')
            metrics[f'eval_ppl_{code_type}_tpl'] = output.metrics['eval_perplexity']

        return metrics

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Compute perplexity instead of loss for evaluation.

        Returns:
            (perplexity, None, None)  # Not loss
        """
        outputs = model(**inputs)
        perplexity = outputs.perplexity  # Median perplexity from model

        return (perplexity, None, None)
```

**Key Feature:** Evaluates **spatial and temporal perplexity** for each code type separately. Returns **median perplexity** (not mean) for robustness.

---

### 7. `promptehr/promptehr.py`

**Purpose:** User-facing API

#### Class: `PromptEHR`

```python
class PromptEHR:
    def fit(self, train_data, val_data):
        """
        Train model on data.

        Args:
            train_data: SequencePatient with training records
            val_data: SequencePatient with validation records
        """
        # 1. Create tokenizers from train data
        self.data_tokenizer = self._create_tokenizers(train_data)

        # 2. Build model
        self.model = self._build_model()

        # 3. Train with HuggingFace Trainer
        trainer = PromptEHRTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.train_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()

    def predict(self, test_data, n_per_sample=1, n=None, sample_config=None, verbose=True):
        """
        Generate synthetic patient records.

        Args:
            test_data: SequencePatient with test records (for structure)
            n_per_sample: Number of synthetic records per test patient
            n: Total number of synthetic records to generate
            sample_config: Generation hyperparameters (temperature, top_k, etc.)

        Returns:
            SequencePatient with synthetic records
        """
        # Prepare generation inputs (initial codes + features)
        dataloader = self._get_test_dataloader(test_data)

        # Generate visit-by-visit, code-type-by-code-type
        synthetic_records = self._predict_on_dataloader(
            dataloader, n, n_per_sample, verbose
        )

        return SequencePatient(synthetic_records, test_data.config)

    def _generation_loop(self, data, inputs):
        """
        Main generation loop.

        For each visit:
            For each code_type (diag, proc, med, ...):
                1. Randomly mask ~50% of real codes (prompts)
                2. Call model.generate() with prompts
                3. Sample new codes to fill remaining slots
                4. Append to sequence
            Add </s> after visit
        """
        # Detailed implementation in codebase
```

**Generation Strategy:**
1. **Visit-by-visit:** Generate one visit at a time
2. **Code-type-by-code-type:** Within each visit, generate diag, then proc, then med
3. **Partial prompting:** Keep ~50% of real codes, generate remaining
4. **Sequential conditioning:** Each code type conditions on previous code types in same visit

---

### 8. `promptehr/generator.py`

**Purpose:** Generation mixin with custom sampling logic

#### Class: `EHRGenerationMixin`

```python
class EHRGenerationMixin:
    def generate(self, input_ids, max_length=None, do_sample=True,
                 temperature=1.0, top_k=None, repetition_penalty=1.0,
                 no_repeat_ngram_size=1, **model_kwargs):
        """
        Generate tokens with sampling (no beam search support).

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            do_sample: Must be True (greedy/beam not supported)
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling (default: None)
            no_repeat_ngram_size: Suppress n-gram repeats (default: 1)

        Returns:
            Generated token IDs [batch, max_length]
        """
        # Custom implementation of sampling loop
        # Converts generated IDs back to code-specific IDs using model_tokenizer

    def sample(self, input_ids, logits_processor=None, stopping_criteria=None,
               logits_warper=None, max_length=None, **model_kwargs):
        """
        Multinomial sampling generation.

        Flow:
            1. Get model logits
            2. Apply logits processors (repetition penalty, etc.)
            3. Apply logits warpers (temperature, top-k)
            4. Sample from distribution
            5. Append to sequence
            6. Repeat until stopping criteria met
        """
```

**Key Feature:** Converts token IDs back to code-specific IDs using `model_tokenizer`, handling unknown tokens gracefully (maps to `<unk>`).

---

### 9. `promptehr/evaluator.py`

**Purpose:** Standalone evaluator for perplexity

#### Class: `Evaluator`

```python
class Evaluator:
    def evaluate(self, code_type, ppl_type='spl', eval_batch_size=8):
        """
        Compute perplexity for given code type.

        Args:
            code_type: 'diag', 'proc', 'med', etc.
            ppl_type: 'spl' (spatial) or 'tpl' (temporal)
            eval_batch_size: Batch size for evaluation

        Returns:
            Median perplexity value
        """
        self.collate_fn.set_eval_code_type(code_type)
        self.collate_fn.set_eval_ppl_type(ppl_type)

        perplexities = []
        for batch in dataloader:
            outputs = self.model(**batch)
            perplexities.append(outputs.perplexity.item())

        return np.median(perplexities)
```

---

## Architecture Details

### Conditional Prompt Injection

**Reparameterization Architecture:**

```
Numerical Features (e.g., age):
    x_num [batch, n_num] → Linear(d_hidden) → ReLU → Linear(d_model) → [batch, n_num, d_model]

Categorical Features (e.g., sex, race):
    x_cat [batch, n_cat] → Embedding → Linear(d_hidden) → ReLU → Linear(d_model) → [batch, n_cat, d_model]

Combined Prompt:
    concat([num_embeds, cat_embeds]) → [batch, n_num+n_cat, d_model]

Prepend to Input:
    [prompt_embeds || input_embeds] → [batch, n_prompts+seq_len, d_model]
```

**Why Reparameterization?**
- Provides flexibility in prompt embedding space
- Hidden layer (d=128) learns feature interactions
- Output layer (d=768) matches BART embedding dimension

---

### Model Architecture Diagram

```
Input Sequence:
    <s> <diag> code1 code2 </diag> <proc> code3 </proc> </s> <s> <diag> code4 </diag> </s>

Features:
    x_num = [age]           → [batch, 1]
    x_cat = [sex, race]     → [batch, 2]

┌────────────────────────────────────────────────────────────┐
│                    ConditionalPrompt                        │
│  x_num + x_cat → prompt_embeds [batch, 3, 768]            │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                   PromptBartEncoder                         │
│  1. Embed input tokens → [batch, seq_len, 768]            │
│  2. Prepend prompt_embeds → [batch, 3+seq_len, 768]       │
│  3. Positional embeddings                                  │
│  4. 6 encoder layers (self-attention + FFN)                │
│  Output: encoder_hidden_states [batch, 3+seq_len, 768]    │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                   PromptBartDecoder                         │
│  1. Shift labels → decoder_input_ids                       │
│  2. Embed decoder input → [batch, tgt_len, 768]           │
│  3. Prepend prompt_embeds → [batch, 3+tgt_len, 768]       │
│  4. 6 decoder layers (self-attn + cross-attn + FFN)        │
│  Output: decoder_hidden_states [batch, 3+tgt_len, 768]    │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│              Code-Type-Specific LM Heads                    │
│  lm_heads['diag']: Linear(768 → vocab_size_diag)          │
│  lm_heads['proc']: Linear(768 → vocab_size_proc)          │
│  lm_heads['med']:  Linear(768 → vocab_size_med)           │
│  Output: logits [batch, tgt_len, vocab_size_code_type]    │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│                    Loss Computation                         │
│  CrossEntropyLoss(logits, encoded_labels)                  │
│  Perplexity = median(exp(-log(prob)))  [if label_mask]    │
└────────────────────────────────────────────────────────────┘
```

---

## Data Processing Pipeline

### Training Data Flow

```
MIMIC-III Raw Data (.jsonl)
  - Format: {"pid": ..., "diag": [[...], [...]], "proc": [[...]], "x_num": [...], "x_cat": [...]}
         ↓
MimicDataset.__getitem__()
  - Returns: {'visits': {...}, 'features': [...], 'pid': ...}
         ↓
MimicDataCollator.__call__()
  ┌─────────────────────────────────────────┐
  │ 1. Randomly select code_type to augment │
  │ 2. For selected code_type:              │
  │    - Span masking (Poisson λ=3)         │
  │    - Token deletion (p=0.15)            │
  │    - Token replacement (p=0.15)         │
  │ 3. For other code_types:                │
  │    - Keep as-is or apply light augment  │
  │ 4. Build next-span prediction task      │
  │ 5. Tokenize to input_ids + labels       │
  └─────────────────────────────────────────┘
         ↓
Batch Dictionary
  {
    'input_ids': [batch, max_len],           # With <mask> tokens
    'attention_mask': [batch, max_len],
    'decoder_input_ids': [batch, max_len],   # Shifted labels
    'labels': [batch, max_len],              # Targets (-100 for padding)
    'label_mask': [batch, max_len],          # 1 for masked positions
    'x_num': [batch, n_num],
    'x_cat': [batch, n_cat],
    'code_type': 'diag' (or 'proc', 'med')
  }
         ↓
BartForEHRSimulation.forward()
  - Computes loss + perplexity
         ↓
Backpropagation + Optimizer Step
```

### Generation Data Flow

```
Test Patient (SequencePatient)
  - Real patient record: visits + features
         ↓
PromptEHR._prepare_input_for_generation()
  ┌─────────────────────────────────────────┐
  │ 1. Sample initial code from first visit │
  │ 2. Build: <s> <diag> init_code ...      │
  │ 3. Extract features: x_num, x_cat       │
  └─────────────────────────────────────────┘
         ↓
PromptEHR._generation_loop()
  ┌─────────────────────────────────────────────────┐
  │ For each visit:                                  │
  │   For each code_type (diag, proc, med):          │
  │     1. Randomly mask ~50% of real codes (prompt) │
  │     2. Encode prompt: <code_type> code1 code2... │
  │     3. Call model.generate():                    │
  │        - Encode input + prompts                  │
  │        - Decode autoregressively                 │
  │        - Sample with temperature/top-k           │
  │        - Stop at </code_type>                    │
  │     4. Extract generated codes                   │
  │     5. Combine: real_prompts + generated_new     │
  │     6. Append: <code_type> all_codes </code_type>│
  │   Append: </s>                                   │
  └─────────────────────────────────────────────────┘
         ↓
Synthetic Patient Record
  {
    'diag': [[code1, code2, ...], [code3, ...]],
    'proc': [[code4, ...]],
    'med': [[code5, code6, ...]],
    'x_num': [age],
    'x_cat': [sex, race]
  }
         ↓
Convert to SequencePatient format
```

---

## Evaluation Framework

### Perplexity Types

#### 1. Spatial Perplexity (SPL)

**Definition:** Perplexity of predicting codes within a visit when other code types are known.

**Setup:**
```python
# Mask all codes of selected code_type
Input:  <s> <diag> <mask> <mask> </diag> <proc> code1 code2 </proc> </s>
Label:  <s> <diag> code3 code4  </diag> <proc> -100  -100   </proc> </s>
        (Only compute loss on <diag> positions)
```

**Interpretation:** Measures how well the model predicts codes given:
- Demographics (x_num, x_cat)
- Other code types in same visit
- Temporal context (previous visits)

#### 2. Temporal Perplexity (TPL)

**Definition:** Perplexity of predicting next visit's codes given previous visits.

**Setup:**
```python
# Mask next visit entirely
Input:  <s> <diag> code1 code2 </diag> </s> <s> <diag> <mask> <mask> </diag> </s>
Label:  <s> <diag> -100  -100  </diag> </s> <s> <diag> code3  code4  </diag> </s>
        (Only compute loss on second visit)
```

**Interpretation:** Measures how well the model predicts future visits given:
- Demographics
- Previous visit codes

---

### Evaluation Metrics Reported

**From codebase analysis, PromptEHR reports:**

| Metric | Code Type | Perplexity Type | Example Value |
|--------|-----------|----------------|---------------|
| `eval_ppl_diag_spl` | Diagnosis | Spatial | 2.34 |
| `eval_ppl_diag_tpl` | Diagnosis | Temporal | 3.12 |
| `eval_ppl_proc_spl` | Procedure | Spatial | 2.89 |
| `eval_ppl_proc_tpl` | Procedure | Temporal | 3.45 |
| `eval_ppl_med_spl` | Medication | Spatial | 2.56 |
| `eval_ppl_med_tpl` | Medication | Temporal | 3.22 |

**Total:** 2 perplexity types × N code types metrics

**Interpretation:**
- Lower perplexity = Better prediction quality
- Spatial perplexity typically < Temporal perplexity (easier task)

---

### What PromptEHR Does NOT Evaluate

❌ **Semantic Coherence:**
- No JS divergence on code frequencies
- No KL divergence
- No distribution matching (KS tests)

❌ **Co-occurrence Patterns:**
- No pairwise co-occurrence analysis
- No code pair frequency comparison
- No graph-based metrics

❌ **Medical Validity:**
- No age-appropriateness checks
- No sex-appropriateness checks
- No ICD-9 validity verification

❌ **Statistical Fidelity:**
- No top-k code overlap
- No Jaccard similarity
- No code frequency histograms

❌ **Diversity:**
- No self-BLEU
- No distinct-n metrics
- No uniqueness measures

**Conclusion:** PromptEHR focuses exclusively on **perplexity-based evaluation** as a proxy for generation quality. They do not implement comprehensive semantic coherence or medical validity checks.

---

## Generation Procedure

### Visit-by-Visit Generation Algorithm

```python
def _generation_loop(data, inputs):
    """
    Generate synthetic patient visits sequentially.

    Strategy:
        - Visit-by-visit generation (not all visits at once)
        - Code-type-by-code-type within each visit
        - Partial prompting (~50% real codes kept)
    """

    generated_visits = {code_type: [] for code_type in code_types}
    current_sequence = inputs['input_ids']  # Initial: <s> <diag> init_code

    for visit_idx in range(num_visits):
        visit_codes = {code_type: [] for code_type in code_types}

        for code_type in code_types:
            # Step 1: Get real codes for this visit + code_type
            real_codes = data['visits'][code_type][visit_idx]

            # Step 2: Randomly mask ~50% (binomial sampling)
            prompt_codes = random_mask(real_codes, keep_prob=0.5)

            # Step 3: Encode prompt
            prompt_tokens = tokenize(f"<{code_type}> {prompt_codes} ...")
            current_sequence = torch.cat([current_sequence, prompt_tokens])

            # Step 4: Generate until </{code_type}>
            generated_ids = model.generate(
                input_ids=current_sequence,
                x_num=data['x_num'],
                x_cat=data['x_cat'],
                max_length=current_length + max_codes_per_type,
                eos_token_id=tokenizer.convert_tokens_to_ids(f"</{code_type}>"),
                temperature=0.7,
                top_k=40,
                no_repeat_ngram_size=1  # Prevent duplicates
            )

            # Step 5: Extract generated codes
            new_codes = extract_codes_between_markers(generated_ids, code_type)

            # Step 6: Combine prompt + generated
            all_codes = prompt_codes + new_codes
            visit_codes[code_type] = all_codes

            # Step 7: Update sequence
            current_sequence = generated_ids

        # Add end-of-visit token
        current_sequence = torch.cat([current_sequence, [eos_token_id]])

        # Store visit
        for code_type in code_types:
            generated_visits[code_type].append(visit_codes[code_type])

    return generated_visits
```

**Key Features:**
1. **Sequential conditioning:** Each code type in a visit conditions on previous code types
2. **Partial prompting:** ~50% of real codes provided as prompts (not zero-shot)
3. **No-repeat n-gram:** Prevents duplicate codes within a visit
4. **Temperature sampling:** Balanced diversity vs quality (default: 0.7)

---

## Comparison: PromptEHR vs Our Implementation

### Architecture Comparison

| Component | PromptEHR | Our Implementation | Match? |
|-----------|-----------|-------------------|--------|
| **Base Model** | BART-base (6L-6L-768H) | BART-base (6L-6L-768H) | ✅ |
| **Conditional Prompts** | x_num + x_cat → reparameterization | x_num + x_cat → reparameterization | ✅ |
| **Prompt Injection** | Prepend to embeddings | Prepend to embeddings | ✅ |
| **Code Vocabularies** | Separate (diag/proc/med) | Single (diag only) | ❌ |
| **LM Heads** | One per code type | One (diagnosis) | ❌ |
| **Training Tasks** | Masking + deletion + replacement | Masking + deletion + replacement | ✅ |
| **Auxiliary Tasks** | None | Age + sex prediction | ❌ |

### Training Procedure Comparison

| Aspect | PromptEHR | Our Implementation | Match? |
|--------|-----------|-------------------|--------|
| **Augmentation** | Span mask + del + rep | Span mask + del + rep + shuffling | ~✅ |
| **Loss Function** | Cross-entropy only | CE + age_loss + sex_loss | ❌ |
| **Optimization** | AdamW | AdamW | ✅ |
| **Learning Rate** | ~1e-4 | 1e-4 | ✅ |
| **Warmup Ratio** | 0.06 | Variable | ~✅ |
| **Batch Size** | 8-16 | 8 | ✅ |
| **Epochs** | Variable | 30 | ~✅ |
| **Model Selection** | Lowest perplexity | Lowest val loss | ~✅ |

### Generation Procedure Comparison

| Aspect | PromptEHR | Our Implementation | Match? |
|--------|-----------|-------------------|--------|
| **Initialization** | Sample real code | Demographics only | ❌ |
| **Prompting** | ~50% real codes | 0% (zero-prompt) OR 0-100% (conditional) | ❌ |
| **Structure** | Visit-by-visit | Visit-by-visit OR single-shot | ~✅ |
| **Code Types** | Sequential (diag→proc→med) | Single (diag only) | ❌ |
| **Stopping** | </code_type> tokens | <END> token (often fails) | ❌ |
| **Sampling** | Temperature + top-k | Temperature + top-k + top-p | ~✅ |
| **No-repeat** | n=1 | n=1 | ✅ |

### Evaluation Metrics Comparison

| Metric | PromptEHR | Our Implementation | Novel? |
|--------|-----------|-------------------|--------|
| **Perplexity (Spatial)** | ✅ | ✅ | ❌ |
| **Perplexity (Temporal)** | ✅ | ✅ | ❌ |
| **Cross-Entropy Loss** | ✅ | ✅ | ❌ |
| **Jaccard Similarity** | ❌ | ✅ | **✅ Novel** |
| **JS Divergence (Code Freq)** | ❌ | ✅ | **✅ Novel** |
| **Co-occurrence Score** | ❌ | ✅ | **✅ Novel** |
| **Top-100 Overlap** | ❌ | ✅ | **✅ Novel** |
| **KS Test (Distributions)** | ❌ | ✅ | **✅ Novel** |
| **Age Appropriateness** | ❌ | ✅ | **✅ Novel** |
| **Sex Appropriateness** | ❌ | ✅ | **✅ Novel** |

**Summary:**
- PromptEHR: **3 metrics** (perplexity variants + loss)
- Our Implementation: **10 metrics** (7 novel)
- **Novel Contribution:** Comprehensive semantic coherence + medical validity evaluation framework

---

## Data Flow Diagrams

### Training Flow (PromptEHR)

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LOADING                            │
└─────────────────────────────────────────────────────────────┘
MIMIC-III .jsonl Files
    ├─ train.jsonl (70%)
    ├─ val.jsonl (15%)
    └─ test.jsonl (15%)
         ↓
MimicDataset
    - Parses JSON to dict format
    - Stores: {visits: {...}, features: [...], pid: ...}
         ↓

┌─────────────────────────────────────────────────────────────┐
│                   BATCH COLLATION                            │
└─────────────────────────────────────────────────────────────┘
MimicDataCollator.call_train()
    ┌────────────────────────────────────┐
    │ For each sample in batch:          │
    │ 1. Select code_type to augment     │
    │ 2. Apply span masking:             │
    │    - Sample spans ~ Poisson(λ=3)   │
    │    - Replace with <mask>           │
    │ 3. Apply deletion (p=0.15)         │
    │ 4. Apply replacement (p=0.15)      │
    │ 5. Build next-span prediction      │
    │ 6. Tokenize                        │
    └────────────────────────────────────┘
         ↓
Batch Tensors
    {
        'input_ids': [8, 512],        # With <mask> tokens
        'labels': [8, 512],           # Targets
        'label_mask': [8, 512],       # 1=masked positions
        'x_num': [8, n_num],
        'x_cat': [8, n_cat],
        'code_type': 'diag'
    }
         ↓

┌─────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                              │
└─────────────────────────────────────────────────────────────┘
BartForEHRSimulation.forward()
    │
    ├─ ConditionalPrompt(x_num, x_cat)
    │  └─> prompt_embeds [8, 3, 768]
    │
    ├─ PromptBartModel.forward()
    │  │
    │  ├─ PromptBartEncoder
    │  │  ├─ Embed input_ids → [8, 512, 768]
    │  │  ├─ Prepend prompts → [8, 515, 768]
    │  │  └─ 6 encoder layers → encoder_hidden_states
    │  │
    │  └─ PromptBartDecoder
    │     ├─ Embed decoder_input → [8, 512, 768]
    │     ├─ Prepend prompts → [8, 515, 768]
    │     ├─ 6 decoder layers (cross-attn to encoder)
    │     └─ decoder_hidden_states [8, 512, 768]
    │
    ├─ LM_head[code_type](decoder_hidden_states)
    │  └─> logits [8, 512, vocab_size_diag]
    │
    ├─ CrossEntropyLoss(logits, labels)
    │  └─> loss (scalar)
    │
    └─ Perplexity(logits[label_mask], labels[label_mask])
       └─> perplexity (median, scalar)
         ↓
Outputs
    {
        'loss': 2.34,
        'perplexity': 10.38,
        'logits': [8, 512, vocab_size],
        ...
    }
         ↓
Backpropagation + Optimizer Step
```

### Generation Flow (PromptEHR)

```
┌─────────────────────────────────────────────────────────────┐
│                  GENERATION INITIALIZATION                   │
└─────────────────────────────────────────────────────────────┘
Test Patient (Real EHR)
    {
        'diag': [[code1, code2], [code3]],
        'proc': [[code4]],
        'med': [[code5, code6]],
        'x_num': [65.0],
        'x_cat': [0, 2]  # M, White
    }
         ↓
_prepare_input_for_generation()
    ├─ Sample init code: code1 (from first visit, first code type)
    ├─ Build: <s> <diag> code1
    └─ Extract features: x_num=[65.0], x_cat=[0,2]
         ↓
Initial State
    {
        'input_ids': [1, 3, 7],  # <s> <diag> code1
        'x_num': [65.0],
        'x_cat': [0, 2]
    }
         ↓

┌─────────────────────────────────────────────────────────────┐
│                    GENERATION LOOP                           │
└─────────────────────────────────────────────────────────────┘
For visit_idx in range(num_visits):  # e.g., 2 visits
    │
    For code_type in ['diag', 'proc', 'med']:
        │
        ├─ Get real codes: [code1, code2] (if visit_idx=0, code_type='diag')
        │
        ├─ Random mask ~50%: keep [code1], mask [code2]
        │
        ├─ Append to input: <s> <diag> code1 code1 ...
        │
        ├─ model.generate()
        │  ├─ Input: [1, 3, 7, 7, ...]
        │  ├─ Features: x_num, x_cat
        │  ├─ Max length: current_length + 20
        │  ├─ EOS: </diag> (token_id=4)
        │  ├─ Temperature: 0.7
        │  ├─ Top-k: 40
        │  ├─ No-repeat n-gram: 1
        │  └─ Sample autoregressively until </diag>
        │     ↓
        │  Generated: [1, 3, 7, 7, 12, 34, 4]
        │             <s> <diag> code1 code1 codeX codeY </diag>
        │
        ├─ Extract new codes: [code1, codeX, codeY]
        │
        ├─ Store: visit[0]['diag'] = [code1, codeX, codeY]
        │
        └─ Update input: [1, 3, 7, 7, 12, 34, 4]
        │
    (Repeat for 'proc', 'med')
    │
    Append </s>: [1, 3, 7, 7, 12, 34, 4, 5, 8, ..., 2]
    │
    Store visit_0
         ↓
Synthetic Patient
    {
        'diag': [[code1, codeX, codeY], [code3, codeZ]],
        'proc': [[code4, codeW]],
        'med': [[code5, code6, codeV]],
        'x_num': [65.0],
        'x_cat': [0, 2]
    }
         ↓
Format to SequencePatient
```

---

## Key Insights

### 1. Perplexity-Only Evaluation

**Finding:** PromptEHR uses **perplexity as the sole quality metric**.

**Implications:**
- No direct measurement of code frequency distributions
- No co-occurrence pattern validation
- No medical validity checks

**Our Contribution:** Comprehensive evaluation framework that measures:
- Statistical fidelity (JS divergence, KS tests)
- Clinical plausibility (co-occurrence, medical validity)
- Generation diversity (top-k overlap, Jaccard)

---

### 2. Partial Prompting vs Zero-Prompt

**Finding:** PromptEHR generates with **~50% real codes as prompts**.

**Difference from our approach:**
- PromptEHR: Reconstruction task (fill in missing codes)
- Ours: Generation task (create from demographics only)

**Implication:** PromptEHR has an **easier task** (given partial ground truth), but we evaluate **harder task** (zero-shot generation).

---

### 3. Multi-Code-Type Structure

**Finding:** PromptEHR uses **separate vocabularies per code type** (diag, proc, med).

**Advantages:**
- Prevents implausible cross-modality combinations
- Smaller vocabulary per LM head (easier learning)
- Natural structure enforcement

**Our simplification:** Single diagnosis code vocabulary (5,562 codes).

**Implication:** We have a **harder sparsity problem** but simpler architecture.

---

### 4. No Hierarchical Generation

**Finding:** PromptEHR uses **flat code vocabularies** (no ICD-9 hierarchy).

**Implications:**
- Suffers from same sparsity problems we face
- Does not leverage medical ontology (ICD-9 chapters)
- Potential for improvement (same as our Tier 2.3 recommendation)

---

### 5. Median Perplexity for Robustness

**Finding:** PromptEHR reports **median perplexity**, not mean.

**Rationale:**
- Median more robust to outliers
- Some patients may have very high perplexity (rare conditions)
- Mean would be dominated by outliers

**Our adoption:** We should consider median for our metrics as well.

---

## Novel Contributions in Our Implementation

Based on comprehensive codebase analysis, **these are novel contributions not present in PromptEHR:**

### 1. Semantic Coherence Evaluation ✅
- **JS Divergence** (code frequency distribution matching)
- **Co-occurrence Score** (pairwise code co-occurrence patterns)
- **Top-100 Overlap** (most common codes matching)
- **KS Tests** (visit/code count distribution matching)

### 2. Medical Validity Evaluation ✅
- **Age-Appropriate Diagnosis Rate**
- **Sex-Appropriate Diagnosis Rate**
- **Duplicate Code Detection**

### 3. Multi-Task Learning ✅
- **Age Prediction Auxiliary Loss**
- **Sex Prediction Auxiliary Loss**
- Balancing medical validity with semantic coherence

### 4. Zero-Prompt Generation ✅
- Generation from demographics only (no code prompts)
- Harder task than PromptEHR's reconstruction approach

### 5. Code Shuffling ✅
- Treats codes as unordered sets during training
- Prevents positional bias
- (PromptEHR preserves order from data)

### 6. Comprehensive Evaluation Framework ✅
- 10 metrics vs 3 metrics in PromptEHR
- Medical + statistical + clinical quality assessment

---

## Recommendations for Documentation

When citing our work vs PromptEHR:

**PromptEHR Contributions (Original Paper):**
- Conditional prompt injection architecture
- Multi-code-type generation framework
- Span masking augmentation strategy
- Perplexity-based evaluation

**Our Novel Contributions:**
- **Semantic coherence evaluation framework** (7 new metrics)
- **Medical validity assessment** (age/sex appropriateness)
- **Multi-task learning** for medical validity
- **Zero-prompt generation** capability
- **Comprehensive quality assessment** beyond perplexity

**Explicitly state in papers/docs:**
> "While PromptEHR focuses on perplexity-based evaluation, we extend the evaluation framework to include semantic coherence (code frequency distributions, co-occurrence patterns) and medical validity (age/sex appropriateness). To our knowledge, this is the first comprehensive evaluation of statistical fidelity and clinical plausibility for EHR generation models."

---

## References

**PromptEHR Repository:**
- Location: `/u/jalenj/PromptEHR`
- Version: As of 2025-10-29 analysis
- License: (Check LICENSE file)

**PromptEHR Paper:**
- Wang et al. (2023)
- arXiv:2307.09123 (likely)
- Title: "PromptEHR: Conditional Electronic Health Records Generation with Prompt-based Learning"

**Key Differences Documented:**
- Architecture: Similar (BART + conditional prompts)
- Evaluation: Vastly different (perplexity only vs comprehensive)
- Task: Different (reconstruction vs zero-shot generation)
- Metrics: 7/10 metrics are novel contributions

---

## Appendix: File-by-File Summary

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `constants.py` | ~50 | Global config | CODE_TYPES, SPECIAL_TOKEN_DICT |
| `data.py` | ~400 | Data containers | Voc, SequencePatient |
| `dataset.py` | ~600 | PyTorch datasets | MimicDataset, MimicDataCollator |
| `evaluator.py` | ~100 | Evaluation | Evaluator.evaluate() |
| `generator.py` | ~300 | Generation mixin | EHRGenerationMixin.generate() |
| `model.py` | ~200 | Main model | BartForEHRSimulation |
| `modeling_config.py` | ~250 | Config & tokenizers | DataTokenizer, ModelTokenizer |
| `modeling_promptbart.py` | ~500 | Conditional prompts | ConditionalPrompt, PromptBartEncoder |
| `promptehr.py` | ~700 | User API | PromptEHR.fit(), predict() |
| `trainer.py` | ~200 | HF trainer | PromptEHRTrainer.evaluate() |
| `demo_data.py` | ~50 | Demo utilities | load_synthetic_data() |
| **Total** | **~3,350** | **11 active files** | **50+ functions** |

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Maintainer:** pehr_scratch project
