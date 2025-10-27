# PromptEHR Architecture Diagrams

Visual documentation of the complete PromptEHR system architecture, data flows, and transformations.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Transformation Pipeline](#2-data-transformation-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Prompt Conditioning Mechanism](#4-prompt-conditioning-mechanism)
5. [Training Flow](#5-training-flow)
6. [Vocabulary & Tokenization](#6-vocabulary--tokenization)
7. [Tensor Shapes Reference](#7-tensor-shapes-reference)
8. [Generation Process](#8-generation-process)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PromptEHR System                                 │
│                                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐│
│  │ Phase 1  │──▶│ Phase 2  │──▶│ Phase 3  │──▶│ Phase 4  │──▶│ Phase 5││
│  │   Data   │   │  Model   │   │ Training │   │Generation│   │Evaluate││
│  │   Prep   │   │   Arch   │   │ Pipeline │   │          │   │        ││
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────┘│
│       │              │               │               │              │    │
│       ▼              ▼               ▼               ▼              ▼    │
│  Vocabulary    PromptBART      Trainer.py      Generator       Metrics  │
│  Tokenizer     Encoder/Dec     Optimizer      Autoregressive  Analysis  │
│  Dataset       Prompt Emb      Checkpoints    Sampling        Quality   │
│  Collator      Attention       Validation     Constraints     Stats     │
└─────────────────────────────────────────────────────────────────────────┘

              Input: MIMIC-III CSVs (46K patients, 651K diagnoses)
                                      ▼
              Output: Synthetic EHR sequences with demographics
```

### Component Dependencies

```
vocabulary.py
     ├──▶ code_tokenizer.py
     │         ├──▶ dataset.py
     │         │        └──▶ trainer.py
     │         │                  └──▶ (trained model)
     │         └──▶ generator.py ──────┘
     │
     └──▶ data_loader.py
               └──▶ dataset.py
```

---

## 2. Data Transformation Pipeline

### 2.1 MIMIC-III to PatientRecord

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Raw MIMIC-III Data (CSVs)                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PATIENTS.csv          ADMISSIONS.csv         DIAGNOSES_ICD.csv        │
│  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐         │
│  │ SUBJECT_ID   │     │ HADM_ID      │      │ HADM_ID      │         │
│  │ GENDER       │     │ SUBJECT_ID   │      │ ICD9_CODE    │         │
│  │ DOB          │     │ ADMITTIME    │      │ SEQ_NUM      │         │
│  │ ...          │     │ ETHNICITY    │      │ ...          │         │
│  └──────────────┘     └──────────────┘      └──────────────┘         │
│         │                    │                       │                 │
│         └────────────────────┴───────────────────────┘                 │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────┐                                 │
│                    │ load_mimic_data │                                 │
│                    └─────────────────┘                                 │
│                              │                                          │
│              ┌───────────────┼───────────────┐                         │
│              ▼               ▼               ▼                         │
│        Calculate Age   Group by Visit   Normalize Ethnicity           │
│        (DOB → age)     (HADM_ID)        (multiple → 6 categories)     │
│                              │                                          │
│                              ▼                                          │
├────────────────────────────────────────────────────────────────────────┤
│                        PatientRecord                                    │
├────────────────────────────────────────────────────────────────────────┤
│  subject_id: 12345                                                     │
│  age: 65.0                          ◀── Continuous feature             │
│  gender: "M"  ──────────────────────▶ Encode to: 0                     │
│  ethnicity: "WHITE"  ────────────────▶ Encode to: 0                    │
│  visits: [                                                              │
│    ["401.9", "250.00"],  ◀── Visit 1 (2 codes)                        │
│    ["428.0"]             ◀── Visit 2 (1 code)                         │
│  ]                                                                      │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  to_dict()      │
                     └─────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Dictionary Format                                  │
├────────────────────────────────────────────────────────────────────────┤
│  x_num:  np.array([65.0])           ◀── Shape: [1]                     │
│  x_cat:  np.array([0, 0])           ◀── Shape: [2] [gender, ethnicity]│
│  visits: [["401.9", "250.00"], ["428.0"]]                              │
│  num_visits: 2                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 PatientRecord to Tokens

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DiagnosisCodeTokenizer                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: visits = [["401.9", "250.00"], ["428.0"]]                     │
│                                                                         │
│  Step 1: Vocabulary Lookup                                             │
│  ┌──────────────────────────────────────────┐                         │
│  │  DiagnosisVocabulary                     │                         │
│  │  ─────────────────────────────────────   │                         │
│  │  "401.9"   → vocab_idx: 0                │                         │
│  │  "250.00"  → vocab_idx: 1                │                         │
│  │  "428.0"   → vocab_idx: 2                │                         │
│  └──────────────────────────────────────────┘                         │
│                    │                                                    │
│                    ▼                                                    │
│  Step 2: Add Code Offset (+6 for special tokens)                      │
│  ┌──────────────────────────────────────────┐                         │
│  │  vocab_idx + 6 = token_id                │                         │
│  │  ─────────────────────────────────────   │                         │
│  │  0 + 6 = 6   (401.9)                     │                         │
│  │  1 + 6 = 7   (250.00)                    │                         │
│  │  2 + 6 = 8   (428.0)                     │                         │
│  └──────────────────────────────────────────┘                         │
│                    │                                                    │
│                    ▼                                                    │
│  Step 3: Add Structural Tokens                                         │
│  ┌──────────────────────────────────────────┐                         │
│  │  Visit 1: [3, 6, 7, 4]                   │                         │
│  │           <v> 401.9 250.00 <\v>          │                         │
│  │                                           │                         │
│  │  Visit 2: [3, 8, 4]                      │                         │
│  │           <v> 428.0 <\v>                 │                         │
│  └──────────────────────────────────────────┘                         │
│                    │                                                    │
│                    ▼                                                    │
│  Step 4: Add BOS and END tokens                                        │
│  ┌──────────────────────────────────────────┐                         │
│  │  [1, 3, 6, 7, 4, 3, 8, 4, 5]             │                         │
│  │   │  └──visit1───┘ └visit2┘ │           │                         │
│  │   BOS                       END          │                         │
│  └──────────────────────────────────────────┘                         │
│                                                                         │
│  Output: token_ids = [1, 3, 6, 7, 4, 3, 8, 4, 5]                     │
│          Length: 9 tokens (3 codes + 6 structural)                     │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Batching with EHRDataCollator

```
┌────────────────────────────────────────────────────────────────────────┐
│               EHRDataCollator (Batching & Padding)                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: List of 4 patients from EHRPatientDataset                     │
│  ──────────────────────────────────────────────────────────────────   │
│  Patient 0: x_num=[65.0],  x_cat=[0,0],  tokens=[1,3,6,7,4,5] (6)    │
│  Patient 1: x_num=[42.0],  x_cat=[1,2],  tokens=[1,3,8,9,4,3,10,4,5] (9) │
│  Patient 2: x_num=[78.0],  x_cat=[0,1],  tokens=[1,3,11,4,5] (5)     │
│  Patient 3: x_num=[55.0],  x_cat=[1,0],  tokens=[1,3,12,13,14,4,5] (7)  │
│                                                                         │
│  max_seq_length = 256                                                  │
│                                                                         │
│  Step 1: Stack Demographics (same size, no padding needed)            │
│  ┌────────────────────────────────────────┐                           │
│  │  x_num = [[65.0],                      │                           │
│  │           [42.0],                      │  ◀── Shape: [4, 1]        │
│  │           [78.0],                      │                           │
│  │           [55.0]]                      │                           │
│  │                                         │                           │
│  │  x_cat = [[0, 0],                      │                           │
│  │           [1, 2],                      │  ◀── Shape: [4, 2]        │
│  │           [0, 1],                      │                           │
│  │           [1, 0]]                      │                           │
│  └────────────────────────────────────────┘                           │
│                                                                         │
│  Step 2: Pad Tokens to max_seq_length                                 │
│  ┌────────────────────────────────────────┐                           │
│  │  Patient 0: [1,3,6,7,4,5,0,0,0,...]    │  ◀── 6 real + 250 pad    │
│  │  Patient 1: [1,3,8,9,4,3,10,4,5,0,...] │  ◀── 9 real + 247 pad    │
│  │  Patient 2: [1,3,11,4,5,0,0,0,0,...]   │  ◀── 5 real + 251 pad    │
│  │  Patient 3: [1,3,12,13,14,4,5,0,...]   │  ◀── 7 real + 249 pad    │
│  │                                         │                           │
│  │  Padding token: 0                      │                           │
│  └────────────────────────────────────────┘                           │
│                                                                         │
│  Step 3: Create Attention Mask (1=real, 0=padding)                    │
│  ┌────────────────────────────────────────┐                           │
│  │  Patient 0: [1,1,1,1,1,1,0,0,0,...]    │                           │
│  │  Patient 1: [1,1,1,1,1,1,1,1,1,0,...]  │                           │
│  │  Patient 2: [1,1,1,1,1,0,0,0,0,...]    │                           │
│  │  Patient 3: [1,1,1,1,1,1,1,0,0,...]    │                           │
│  └────────────────────────────────────────┘                           │
│                                                                         │
│  Step 4: Create Labels (padding → -100)                               │
│  ┌────────────────────────────────────────┐                           │
│  │  Same as input_ids but:                │                           │
│  │  • Padding tokens (0) → -100           │                           │
│  │  • CrossEntropyLoss ignores -100       │                           │
│  └────────────────────────────────────────┘                           │
│                                                                         │
│  Output Batch:                                                         │
│  ┌────────────────────────────────────────┐                           │
│  │  x_num:          [4, 1]                │                           │
│  │  x_cat:          [4, 2]                │                           │
│  │  input_ids:      [4, 256]              │                           │
│  │  attention_mask: [4, 256]              │                           │
│  │  labels:         [4, 256]              │                           │
│  └────────────────────────────────────────┘                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture

### 3.1 PromptBartModel Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PromptBartModel                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: x_num [4,1], x_cat [4,2], input_ids [4,256]                   │
│                                                                         │
│         ┌──────────────────────────────────────┐                       │
│         │      ConditionalPrompt               │                       │
│         │  ────────────────────────────────    │                       │
│         │  x_num [4,1]  ──▶ Linear(1→768)     │                       │
│         │                   ▼                   │                       │
│         │              [4, 1, 768]              │                       │
│         │                                       │                       │
│         │  x_cat [4,2] ──▶ Embedding(2→768)    │                       │
│         │                  Embedding(6→768)    │                       │
│         │                   ▼                   │                       │
│         │              [4, 2, 768]              │                       │
│         │                   ▼                   │                       │
│         │         Concatenate dim=1             │                       │
│         │                   ▼                   │                       │
│         │      prompt_embeds [4, 3, 768]       │                       │
│         └──────────────────────────────────────┘                       │
│                         │                                               │
│                         │                                               │
│         ┌───────────────┼───────────────────────┐                      │
│         ▼               ▼                       ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │   Encoder    │  │   Decoder    │  │   LM Head    │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
│         │               │                      │                       │
│         │               │                      │                       │
│  ┌──────▼───────────────▼──────────────────────▼──────┐               │
│  │            Detailed Architecture Below              │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Output: logits [4, 256, vocab_size], loss (scalar)                   │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Encoder Architecture (PromptBartEncoder)

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PromptBartEncoder                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input:                                                                 │
│    input_ids:          [batch, seq_len]           = [4, 256]           │
│    inputs_prompt_embeds: [batch, n_prompts, 768] = [4, 3, 768]        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 1: Token Embedding                             │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  input_ids [4, 256]                                  │             │
│  │       │                                               │             │
│  │       ▼  Embedding(vocab_size → 768)                 │             │
│  │  inputs_embeds [4, 256, 768]                         │             │
│  │       │                                               │             │
│  │       │  * sqrt(768) if scale_embedding=True         │             │
│  │       ▼                                               │             │
│  │  inputs_embeds [4, 256, 768]                         │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Prepend Prompt Embeddings                 │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  inputs_prompt_embeds [4, 3, 768]                  │               │
│  │              +                                      │               │
│  │  inputs_embeds        [4, 256, 768]                │               │
│  │              ▼                                      │               │
│  │  torch.cat([prompts, inputs], dim=1)               │               │
│  │              ▼                                      │               │
│  │  inputs_embeds [4, 3+256=259, 768]                 │               │
│  │                │                                    │               │
│  │  Attention mask also extended:                     │               │
│  │  [4, 256] → [4, 3+256=259]                         │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Positional Embeddings                     │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  embed_positions(inputs_embeds)                    │               │
│  │              ▼                                      │               │
│  │  positions [4, 259, 768]                           │               │
│  │              +                                      │               │
│  │  inputs_embeds [4, 259, 768]                       │               │
│  │              ▼                                      │               │
│  │  hidden_states [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 4: Layer Normalization + Dropout             │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  LayerNorm(hidden_states)                          │               │
│  │              ▼                                      │               │
│  │  Dropout(p=0.1)                                    │               │
│  │              ▼                                      │               │
│  │  hidden_states [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 5: Transformer Layers (6 layers)             │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  for layer in encoder_layers:                      │               │
│  │    ┌────────────────────────────────┐              │               │
│  │    │ Self-Attention                 │              │               │
│  │    │  Q, K, V = linear(hidden)      │              │               │
│  │    │  attn = softmax(QK^T / √d_k)   │              │               │
│  │    │  output = attn @ V              │              │               │
│  │    │  ▼                              │              │               │
│  │    │ [4, 259, 768]                  │              │               │
│  │    └────────────────────────────────┘              │               │
│  │           ▼                                         │               │
│  │    ┌────────────────────────────────┐              │               │
│  │    │ Feed-Forward Network           │              │               │
│  │    │  FFN(x) = Linear(GELU(Linear)) │              │               │
│  │    │  ▼                              │              │               │
│  │    │ [4, 259, 768]                  │              │               │
│  │    └────────────────────────────────┘              │               │
│  │           ▼                                         │               │
│  │  hidden_states [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│                    ▼                                                    │
│  Output: encoder_outputs [4, 259, 768]                                 │
│          (3 prompt vectors + 256 input tokens)                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Decoder Architecture (PromptBartDecoder)

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PromptBartDecoder                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input:                                                                 │
│    decoder_input_ids:    [batch, tgt_len]          = [4, 256]          │
│    encoder_hidden_states: [batch, src_len, 768]    = [4, 259, 768]     │
│    inputs_prompt_embeds:  [batch, n_prompts, 768]  = [4, 3, 768]      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 1: Token Embedding                             │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  decoder_input_ids [4, 256]                          │             │
│  │       │                                               │             │
│  │       ▼  Embedding(vocab_size → 768)                 │             │
│  │  inputs_embeds [4, 256, 768]                         │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Prepend Prompt Embeddings                 │               │
│  │         (ONLY if past_key_values is None)          │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  torch.cat([prompts, inputs], dim=1)               │               │
│  │              ▼                                      │               │
│  │  inputs_embeds [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Positional Embeddings                     │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  positions = embed_positions(inputs_embeds)        │               │
│  │  hidden_states = inputs_embeds + positions         │               │
│  │              ▼                                      │               │
│  │  hidden_states [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 4: Transformer Decoder Layers (6 layers)     │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  for layer in decoder_layers:                      │               │
│  │    ┌────────────────────────────────┐              │               │
│  │    │ Causal Self-Attention          │              │               │
│  │    │  (with causal mask)            │              │               │
│  │    │  Attends only to previous      │              │               │
│  │    │  positions                     │              │               │
│  │    │  ▼                              │              │               │
│  │    │ [4, 259, 768]                  │              │               │
│  │    └────────────────────────────────┘              │               │
│  │           ▼                                         │               │
│  │    ┌────────────────────────────────┐              │               │
│  │    │ Cross-Attention                │              │               │
│  │    │  Q = decoder hidden states     │              │               │
│  │    │  K, V = encoder outputs        │              │               │
│  │    │  (includes prompt info!)       │              │               │
│  │    │  ▼                              │              │               │
│  │    │ [4, 259, 768]                  │              │               │
│  │    └────────────────────────────────┘              │               │
│  │           ▼                                         │               │
│  │    ┌────────────────────────────────┐              │               │
│  │    │ Feed-Forward Network           │              │               │
│  │    │  ▼                              │              │               │
│  │    │ [4, 259, 768]                  │              │               │
│  │    └────────────────────────────────┘              │               │
│  │           ▼                                         │               │
│  │  hidden_states [4, 259, 768]                       │               │
│  └────────────────────────────────────────────────────┘               │
│                    │                                                    │
│                    ▼                                                    │
│  Output: decoder_outputs [4, 259, 768]                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Language Modeling Head

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Language Modeling Head                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: decoder_outputs [4, 259, 768]                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 1: Linear Projection                           │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  lm_head: Linear(768 → vocab_size)                   │             │
│  │       │                                               │             │
│  │       ▼                                               │             │
│  │  logits [4, 259, vocab_size]                         │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Slice Off Prompt Positions                │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  if prompt_embeds is not None:                     │               │
│  │    n_prompts = 3                                   │               │
│  │    logits = logits[:, n_prompts:, :]               │               │
│  │       │                                             │               │
│  │       ▼                                             │               │
│  │  logits [4, 256, vocab_size]                       │               │
│  │                                                     │               │
│  │  Now matches labels shape [4, 256]                 │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Compute Loss                              │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  loss = CrossEntropyLoss(                          │               │
│  │      logits.reshape(-1, vocab_size),               │               │
│  │      labels.view(-1)                               │               │
│  │  )                                                  │               │
│  │       │                                             │               │
│  │       ▼                                             │               │
│  │  loss (scalar)                                     │               │
│  │                                                     │               │
│  │  Note: Labels with -100 are ignored                │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Output: logits [4, 256, vocab_size], loss (scalar)                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Prompt Conditioning Mechanism

### 4.1 ConditionalPrompt Internals

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ConditionalPrompt Module                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Numerical Features (Continuous)                     │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  x_num: [batch, 1]  (Age)                            │             │
│  │       │                                               │             │
│  │       ▼  Linear(1 → 768)                             │             │
│  │       │                                               │             │
│  │  [batch, 768]                                         │             │
│  │       │                                               │             │
│  │       ▼  Reshape                                      │             │
│  │       │                                               │             │
│  │  [batch, 1, 768]  ◀── Age prompt vector              │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Categorical Features (Discrete)                   │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  x_cat: [batch, 2]  (Gender, Ethnicity)            │               │
│  │       │                                             │               │
│  │       ├──────────────┬─────────────┐               │               │
│  │       │              │             │               │               │
│  │       ▼              ▼             ▼               │               │
│  │  Gender ID      Ethnicity ID                       │               │
│  │  [batch]         [batch]                           │               │
│  │       │              │                              │               │
│  │       ▼              ▼                              │               │
│  │  Embedding(2)   Embedding(6)                       │               │
│  │  vocab→768      vocab→768                          │               │
│  │       │              │                              │               │
│  │       ▼              ▼                              │               │
│  │  [batch, 768]   [batch, 768]                       │               │
│  │       │              │                              │               │
│  │       ▼              ▼                              │               │
│  │  [batch,1,768]  [batch,1,768]                      │               │
│  │       └──────────────┴─────────────┐               │               │
│  │                                    │               │               │
│  │                                    ▼               │               │
│  │                        Concatenate dim=1           │               │
│  │                                    │               │               │
│  │                                    ▼               │               │
│  │                        [batch, 2, 768]             │               │
│  │                      Gender + Ethnicity prompts    │               │
│  └─────────────────────────────────────────────────────┘               │
│                                    │                                    │
│  ┌─────────────────────────────────▼────────────────┐                 │
│  │  Combine All Prompts                             │                 │
│  │  ───────────────────────────────────────────     │                 │
│  │  [batch, 1, 768] (Age)                           │                 │
│  │         +                                         │                 │
│  │  [batch, 2, 768] (Gender, Ethnicity)             │                 │
│  │         ▼                                         │                 │
│  │  torch.cat(dim=1)                                │                 │
│  │         ▼                                         │                 │
│  │  [batch, 3, 768]                                 │                 │
│  │   Final prompt embeddings                        │                 │
│  └──────────────────────────────────────────────────┘                 │
│                                                                         │
│  Output: prompt_embeds [batch, 3, 768]                                 │
│          3 prompt vectors = 1 age + 2 categorical                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 How Prompts Flow Through Model

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Prompt Flow During Training                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Demographics: x_num=[65.0], x_cat=[0, 0] (65yo, Male, White)         │
│                     │                                                   │
│                     ▼                                                   │
│           ConditionalPrompt                                             │
│                     │                                                   │
│                     ▼                                                   │
│          prompt_embeds [1, 3, 768]                                      │
│                     │                                                   │
│      ┌──────────────┴──────────────┐                                   │
│      │                              │                                   │
│      ▼                              ▼                                   │
│  ┌────────┐                    ┌────────┐                              │
│  │Encoder │                    │Decoder │                              │
│  └────────┘                    └────────┘                              │
│      │                              │                                   │
│  ┌───▼────────────────┐      ┌─────▼──────────────┐                   │
│  │ Prepend to input:  │      │ Prepend to decoder:│                   │
│  │ [3,768] + [256,768]│      │ [3,768] + [256,768]│                   │
│  │        ▼           │      │        ▼           │                   │
│  │   [259, 768]       │      │   [259, 768]       │                   │
│  │                    │      │                    │                   │
│  │ ┌───┬───┬───┬────┐│      │ ┌───┬───┬───┬────┐│                   │
│  │ │Age│Gen│Eth│Seq ││      │ │Age│Gen│Eth│Seq ││                   │
│  │ └───┴───┴───┴────┘│      │ └───┴───┴───┴────┘│                   │
│  │   3      256       │      │   3      256       │                   │
│  └────────────────────┘      └────────────────────┘                   │
│           │                           │                                 │
│           ▼                           ▼                                 │
│  Self-Attention sees:        Cross-Attention:                          │
│  • Prompt embeddings         • Queries from decoder                    │
│  • Input tokens              • Keys/Values from encoder                │
│  • Learns correlations       • (which include prompt info)             │
│                                                                         │
│  Result: Model learns that:                                            │
│  • Age 65+ → more likely chronic conditions (HTN, DM)                  │
│  • Male → different disease patterns than Female                       │
│  • Ethnicity → population-specific disease prevalence                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Flow

### 5.1 Single Training Step

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Single Training Step                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 1: Get Batch from DataLoader                   │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  batch = next(train_loader)                          │             │
│  │       │                                               │             │
│  │       ▼                                               │             │
│  │  x_num:          [16, 1]                             │             │
│  │  x_cat:          [16, 2]                             │             │
│  │  input_ids:      [16, 256]                           │             │
│  │  attention_mask: [16, 256]                           │             │
│  │  labels:         [16, 256]                           │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Move to GPU                               │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  .to(device='cuda')                                │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Forward Pass                              │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  outputs = model(                                  │               │
│  │      input_ids=input_ids,                          │               │
│  │      attention_mask=attention_mask,                │               │
│  │      labels=labels,                                │               │
│  │      x_num=x_num,                                  │               │
│  │      x_cat=x_cat                                   │               │
│  │  )                                                  │               │
│  │       │                                             │               │
│  │       ├──▶ prompt_embeds [16, 3, 768]              │               │
│  │       ├──▶ encoder_outputs [16, 259, 768]          │               │
│  │       ├──▶ decoder_outputs [16, 259, 768]          │               │
│  │       ├──▶ logits [16, 256, vocab_size]            │               │
│  │       └──▶ loss (scalar)                           │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 4: Backward Pass                             │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  loss.backward()                                   │               │
│  │       │                                             │               │
│  │       ▼  Compute gradients via backprop            │               │
│  │       │                                             │               │
│  │  ∇loss/∂θ for all parameters θ                     │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 5: Gradient Clipping                         │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  torch.nn.utils.clip_grad_norm_(                   │               │
│  │      model.parameters(),                           │               │
│  │      max_norm=1.0                                  │               │
│  │  )                                                  │               │
│  │       │                                             │               │
│  │       ▼  Prevents exploding gradients              │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 6: Optimizer Step                            │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  optimizer.step()                                  │               │
│  │       │                                             │               │
│  │       ▼  Update parameters                         │               │
│  │       │                                             │               │
│  │  θ_new = θ_old - lr * ∇loss/∂θ                     │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 7: Scheduler Step                            │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  scheduler.step()                                  │               │
│  │       │                                             │               │
│  │       ▼  Adjust learning rate                      │               │
│  │       │                                             │               │
│  │  lr = f(step, warmup, total_steps)                 │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 8: Zero Gradients                            │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  optimizer.zero_grad()                             │               │
│  │       │                                             │               │
│  │       ▼  Clear for next step                       │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Repeat for all batches in epoch                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Complete Training Loop

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Complete Training Loop                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Initialization:                                                        │
│  ──────────────                                                         │
│  • Load MIMIC-III data (3000 patients)                                 │
│  • Create vocabulary (~8000 codes)                                     │
│  • Initialize PromptBartModel (~101M params)                           │
│  • Create optimizer (AdamW, lr=1e-4)                                   │
│  • Create scheduler (linear warmup)                                    │
│  • Split data (80% train, 20% val)                                     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  For epoch in range(num_epochs):                     │             │
│  │                                                       │             │
│  │    ┌───────────────────────────────────┐             │             │
│  │    │  Training Phase                   │             │             │
│  │    │  ────────────────────────────     │             │             │
│  │    │  model.train()                    │             │             │
│  │    │                                   │             │             │
│  │    │  for batch in train_loader:      │             │             │
│  │    │    • Forward pass                 │             │             │
│  │    │    • Compute loss                 │             │             │
│  │    │    • Backward pass                │             │             │
│  │    │    • Clip gradients               │             │             │
│  │    │    • Optimizer step               │             │             │
│  │    │    • Scheduler step               │             │             │
│  │    │    • Log metrics every N steps    │             │             │
│  │    │                                   │             │             │
│  │    │  Metrics:                         │             │             │
│  │    │  • Average loss per epoch         │             │             │
│  │    │  • Perplexity (exp(loss))         │             │             │
│  │    │  • Learning rate                  │             │             │
│  │    └───────────────────────────────────┘             │             │
│  │                  │                                    │             │
│  │    ┌─────────────▼─────────────────────┐             │             │
│  │    │  Validation Phase                 │             │             │
│  │    │  ────────────────────────────     │             │             │
│  │    │  model.eval()                     │             │             │
│  │    │                                   │             │             │
│  │    │  with torch.no_grad():            │             │             │
│  │    │    for batch in val_loader:      │             │             │
│  │    │      • Forward pass               │             │             │
│  │    │      • Compute loss               │             │             │
│  │    │      • Track metrics              │             │             │
│  │    │                                   │             │             │
│  │    │  Metrics:                         │             │             │
│  │    │  • Validation loss                │             │             │
│  │    │  • Validation perplexity          │             │             │
│  │    │  • Token accuracy                 │             │             │
│  │    │  • Code accuracy                  │             │             │
│  │    └───────────────────────────────────┘             │             │
│  │                  │                                    │             │
│  │    ┌─────────────▼─────────────────────┐             │             │
│  │    │  Checkpointing                    │             │             │
│  │    │  ────────────────────────────     │             │             │
│  │    │  if val_loss < best_val_loss:     │             │             │
│  │    │    • Save best model              │             │             │
│  │    │    • Update best_val_loss         │             │             │
│  │    │                                   │             │             │
│  │    │  if epoch % save_frequency == 0:  │             │             │
│  │    │    • Save checkpoint              │             │             │
│  │    └───────────────────────────────────┘             │             │
│  │                                                       │             │
│  └───────────────────────────────────────────────────────┘             │
│                                                                         │
│  Expected Progress (30 epochs):                                        │
│  ─────────────────────────────────                                     │
│  Epoch 1:  loss=5.5, perplexity=245                                    │
│  Epoch 5:  loss=4.0, perplexity=55                                     │
│  Epoch 10: loss=3.5, perplexity=33                                     │
│  Epoch 15: loss=3.0, perplexity=20                                     │
│  Epoch 20: loss=2.7, perplexity=15                                     │
│  Epoch 30: loss=2.5, perplexity=12                                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Vocabulary & Tokenization

### 6.1 Token ID Structure

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Token ID Allocation                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────┬──────────────────────────────────┬─────────────────┐  │
│  │  ID Range  │       Token Type                 │  Example        │  │
│  ├────────────┼──────────────────────────────────┼─────────────────┤  │
│  │     0      │  <PAD>    Padding token          │  <PAD>          │  │
│  │     1      │  <BOS>    Beginning of sequence  │  <BOS>          │  │
│  │     2      │  <EOS>    End of sequence (BART) │  <EOS>          │  │
│  │     3      │  <v>      Visit start marker     │  <v>            │  │
│  │     4      │  <\v>     Visit end marker       │  <\v>           │  │
│  │     5      │  <END>    Patient sequence end   │  <END>          │  │
│  ├────────────┼──────────────────────────────────┼─────────────────┤  │
│  │   6-N      │  Medical Codes                   │  6="V3001"      │  │
│  │            │  (N = vocab_size - 1)            │  7="250.00"     │  │
│  │            │                                  │  8="401.9"      │  │
│  │            │                                  │  9="428.0"      │  │
│  │            │                                  │  ...            │  │
│  │            │                                  │  N="999.99"     │  │
│  └────────────┴──────────────────────────────────┴─────────────────┘  │
│                                                                         │
│  Total Vocabulary Size: 6 (special) + ~8000 (codes) = ~8006 tokens    │
│                                                                         │
│  Note: Medical codes are 1:1 mapped (NO FRAGMENTATION)                 │
│        Each code = exactly one token ID                                │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Example Tokenization

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Example: 2-Visit Patient                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Patient:                                                         │
│  ──────────────                                                         │
│  Age: 65, Gender: Male, Ethnicity: White                               │
│  Visit 1: ["V3001", "250.00"]     (newborn, diabetes)                  │
│  Visit 2: ["401.9", "428.0"]      (hypertension, heart failure)        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 1: Vocabulary Lookup                           │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  "V3001"  → vocab_idx: 0  → token_id: 6             │             │
│  │  "250.00" → vocab_idx: 1  → token_id: 7             │             │
│  │  "401.9"  → vocab_idx: 2  → token_id: 8             │             │
│  │  "428.0"  → vocab_idx: 3  → token_id: 9             │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Add Structural Tokens                     │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Visit 1: [<v>, V3001, 250.00, <\v>]              │               │
│  │         = [3,   6,     7,      4]                  │               │
│  │                                                     │               │
│  │  Visit 2: [<v>, 401.9, 428.0, <\v>]               │               │
│  │         = [3,   8,     9,     4]                   │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Add Sequence Tokens                       │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  [<BOS>, <v>, V3001, 250.00, <\v>,                │               │
│  │          <v>, 401.9, 428.0,  <\v>, <END>]         │               │
│  │  ═════════════════════════════════════════════     │               │
│  │  [1, 3, 6, 7, 4, 3, 8, 9, 4, 5]                   │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Result: 10 tokens total                                               │
│  • 1 BOS token                                                          │
│  • 4 visit structure tokens (2 <v> + 2 <\v>)                          │
│  • 4 diagnosis code tokens                                             │
│  • 1 END token                                                          │
│                                                                         │
│  Comparison to Text-based BART tokenization:                           │
│  ────────────────────────────────────────────────                      │
│  Our approach:    10 tokens                                            │
│  BART tokenizer:  ~20-25 tokens (codes get fragmented)                 │
│                                                                         │
│  Efficiency gain: ~50% fewer tokens!                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Tensor Shapes Reference

### 7.1 Training Shapes

```
┌────────────────────────────────────────────────────────────────────────┐
│              Tensor Shapes at Each Stage (Training)                     │
├─────────────────────┬───────────────────────┬──────────────────────────┤
│  Location           │  Tensor               │  Shape                   │
├─────────────────────┼───────────────────────┼──────────────────────────┤
│                                                                         │
│  Input (batch)                                                          │
│  ──────────────────────────────────────────────────────────────────    │
│  DataLoader         │  x_num                │  [batch, 1]              │
│                     │  x_cat                │  [batch, 2]              │
│                     │  input_ids            │  [batch, seq_len]        │
│                     │  attention_mask       │  [batch, seq_len]        │
│                     │  labels               │  [batch, seq_len]        │
│                                                                         │
│  Example with batch=16, seq_len=256:                                   │
│                     │  x_num                │  [16, 1]                 │
│                     │  x_cat                │  [16, 2]                 │
│                     │  input_ids            │  [16, 256]               │
│                     │  attention_mask       │  [16, 256]               │
│                     │  labels               │  [16, 256]               │
│                                                                         │
│  ConditionalPrompt                                                      │
│  ──────────────────────────────────────────────────────────────────    │
│  Numerical          │  x_num                │  [16, 1]                 │
│  embedding          │  after Linear         │  [16, 1, 768]            │
│                                                                         │
│  Categorical        │  x_cat[:,0]           │  [16] (gender)           │
│  embedding          │  after Emb(2,768)     │  [16, 1, 768]            │
│                     │  x_cat[:,1]           │  [16] (ethnicity)        │
│                     │  after Emb(6,768)     │  [16, 1, 768]            │
│                     │  concatenated         │  [16, 2, 768]            │
│                                                                         │
│  Combined           │  prompt_embeds        │  [16, 3, 768]            │
│  prompts            │  (1 num + 2 cat)      │                          │
│                                                                         │
│  Encoder                                                                │
│  ──────────────────────────────────────────────────────────────────    │
│  Token embedding    │  input_ids            │  [16, 256]               │
│                     │  after embed          │  [16, 256, 768]          │
│                                                                         │
│  After prepending   │  prompts + tokens     │  [16, 259, 768]          │
│  prompts            │  (3 + 256)            │                          │
│                                                                         │
│  Attention mask     │  extended             │  [16, 259]               │
│                                                                         │
│  Encoder output     │  hidden_states        │  [16, 259, 768]          │
│                                                                         │
│  Decoder                                                                │
│  ──────────────────────────────────────────────────────────────────    │
│  Input              │  decoder_input_ids    │  [16, 256]               │
│                     │  after embed          │  [16, 256, 768]          │
│                                                                         │
│  After prepending   │  prompts + tokens     │  [16, 259, 768]          │
│  prompts            │                       │                          │
│                                                                         │
│  Decoder output     │  hidden_states        │  [16, 259, 768]          │
│                                                                         │
│  Language Model Head                                                    │
│  ──────────────────────────────────────────────────────────────────    │
│  Before slicing     │  lm_logits            │  [16, 259, vocab_size]   │
│                                                                         │
│  After slicing      │  lm_logits            │  [16, 256, vocab_size]   │
│  (remove prompts)   │  (removed 3 prompts)  │                          │
│                                                                         │
│  Labels             │  labels               │  [16, 256]               │
│                                                                         │
│  Loss               │  loss                 │  scalar                  │
│                                                                         │
└─────────────────────┴───────────────────────┴──────────────────────────┘
```

### 7.2 Generation Shapes

```
┌────────────────────────────────────────────────────────────────────────┐
│              Tensor Shapes During Generation                            │
├─────────────────────┬───────────────────────┬──────────────────────────┤
│  Location           │  Tensor               │  Shape                   │
├─────────────────────┼───────────────────────┼──────────────────────────┤
│                                                                         │
│  Input (single patient)                                                 │
│  ──────────────────────────────────────────────────────────────────    │
│  Demographics       │  x_num                │  [1, 1]                  │
│                     │  x_cat                │  [1, 2]                  │
│                     │  input_ids (encoder)  │  [1, initial_len]        │
│                                                                         │
│  ConditionalPrompt                                                      │
│  ──────────────────────────────────────────────────────────────────    │
│  (computed once)    │  prompt_embeds        │  [1, 3, 768]             │
│                                                                         │
│  Encoder (first call)                                                   │
│  ──────────────────────────────────────────────────────────────────    │
│  With prompts       │  encoder_outputs      │  [1, 3+initial_len, 768] │
│                                                                         │
│  Decoder (autoregressive steps)                                         │
│  ──────────────────────────────────────────────────────────────────    │
│  Step 1:            │  decoder_input_ids    │  [1, 1] (just BOS)       │
│  With prompts       │  after prepend        │  [1, 4, 768]             │
│                     │  (3 prompts + 1 token)│                          │
│                     │  output logits        │  [1, 1, vocab_size]      │
│                     │                       │  (sliced)                │
│                                                                         │
│  Step 2:            │  decoder_input_ids    │  [1, 1] (last token)     │
│  NO prompts         │  (cached)             │                          │
│  (from cache)       │  output logits        │  [1, 1, vocab_size]      │
│                                                                         │
│  Step N:            │  decoder_input_ids    │  [1, 1]                  │
│                     │  output logits        │  [1, 1, vocab_size]      │
│                                                                         │
│  Final generated sequence                                               │
│  ──────────────────────────────────────────────────────────────────    │
│  Generated IDs      │  generated_ids        │  [1, generated_len]      │
│                     │  Example: 20 tokens   │  [1, 20]                 │
│                                                                         │
└─────────────────────┴───────────────────────┴──────────────────────────┘
```

---

## 8. Generation Process

### 8.1 Autoregressive Generation

```
┌────────────────────────────────────────────────────────────────────────┐
│                  Autoregressive Generation Flow                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Demographics (age=65, gender=M, ethnicity=WHITE)               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  Step 0: Initialization                              │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  • Encode demographics → prompt_embeds [1,3,768]     │             │
│  │  • Initialize: generated_ids = [BOS]                 │             │
│  │  • Encoder forward (once): context embeddings        │             │
│  └──────────────────────────────────────────────────────┘             │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 1: Generate First Token                      │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Input to decoder: [BOS] + prompts                 │               │
│  │  │                                                  │               │
│  │  ▼ Model forward pass                              │               │
│  │  │                                                  │               │
│  │  logits [1, 1, vocab_size]                         │               │
│  │  │                                                  │               │
│  │  ▼ Sample next token                               │               │
│  │  │                                                  │               │
│  │  next_token = sample(logits, temp=1.3, top_k=50)   │               │
│  │  │                                                  │               │
│  │  ▼ Likely result: <v> (visit start)                │               │
│  │  │                                                  │               │
│  │  generated_ids = [BOS, <v>]                        │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 2: Generate Diagnosis Code                   │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Input: [BOS, <v>]                                 │               │
│  │  │  (decoder only sees last token due to cache)    │               │
│  │  │                                                  │               │
│  │  ▼ Model forward (with cached KV pairs)            │               │
│  │  │                                                  │               │
│  │  logits [1, 1, vocab_size]                         │               │
│  │  │                                                  │               │
│  │  ▼ Sample (biased by demographics via prompts)     │               │
│  │  │                                                  │               │
│  │  next_token = 6 (401.9 - hypertension)             │               │
│  │  │  Age 65 + Male → high probability HTN           │               │
│  │  │                                                  │               │
│  │  generated_ids = [BOS, <v>, 401.9]                 │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 3: Continue Generating Codes                 │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Input: [BOS, <v>, 401.9]                          │               │
│  │  ▼                                                  │               │
│  │  next_token = 7 (250.00 - diabetes)                │               │
│  │  │  HTN + Age 65 → likely comorbid diabetes        │               │
│  │  │                                                  │               │
│  │  generated_ids = [BOS, <v>, 401.9, 250.00]         │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 4: End Visit                                 │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Input: [BOS, <v>, 401.9, 250.00]                  │               │
│  │  ▼                                                  │               │
│  │  next_token = <\v> (end visit)                     │               │
│  │  │                                                  │               │
│  │  generated_ids = [BOS, <v>, 401.9, 250.00, <\v>]   │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Step 5-N: Generate More Visits                    │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  Repeat steps 1-4 for additional visits            │               │
│  │  Model learns disease progression:                 │               │
│  │  • Visit 1: HTN + Diabetes (initial diagnosis)     │               │
│  │  • Visit 2: Heart failure (consequence)            │               │
│  │  • Visit 3: Kidney disease (further complication)  │               │
│  └─────────────────────────────────────────────────────┘               │
│                    │                                                    │
│  ┌─────────────────▼──────────────────────────────────┐               │
│  │  Final: End Sequence                               │               │
│  │  ──────────────────────────────────────────────    │               │
│  │  next_token = <END>                                │               │
│  │  │                                                  │               │
│  │  Final: [BOS, <v>, 401.9, 250.00, <\v>,            │               │
│  │              <v>, 428.0, 585.6, <\v>, <END>]       │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
│  Output: Synthetic patient sequence with realistic disease progression │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Sampling Strategies

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Sampling Strategies                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: logits [1, 1, vocab_size] from model                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  1. Greedy Sampling (deterministic)                  │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  next_token = argmax(logits)                         │             │
│  │                                                       │             │
│  │  Pros: Deterministic, highest probability            │             │
│  │  Cons: Repetitive, no diversity                      │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  2. Temperature Sampling                             │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  scaled_logits = logits / temperature                │             │
│  │  probs = softmax(scaled_logits)                      │             │
│  │  next_token = sample(probs)                          │             │
│  │                                                       │             │
│  │  Temperature effects:                                │             │
│  │  • temp=1.0: Standard probabilities                  │             │
│  │  • temp<1.0: More confident (peaked distribution)    │             │
│  │  • temp>1.0: More diverse (flat distribution)        │             │
│  │                                                       │             │
│  │  Example (temp=1.3 for EHR):                         │             │
│  │    Balances realism with diversity                   │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  3. Top-k Sampling                                   │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  top_k_logits = select_top_k(logits, k=50)           │             │
│  │  probs = softmax(top_k_logits)                       │             │
│  │  next_token = sample(probs)                          │             │
│  │                                                       │             │
│  │  Limits sampling to top k most likely tokens         │             │
│  │  Prevents sampling very unlikely codes               │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  4. Top-p (Nucleus) Sampling                         │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  sorted_probs = sort(softmax(logits))                │             │
│  │  cumsum = cumulative_sum(sorted_probs)               │             │
│  │  nucleus = where(cumsum <= p)                        │             │
│  │  next_token = sample(nucleus)                        │             │
│  │                                                       │             │
│  │  Samples from smallest set with cumulative prob >= p │             │
│  │  Adaptive: set size changes with distribution        │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐             │
│  │  5. Constrained Sampling (Future)                    │             │
│  │  ───────────────────────────────────────────────     │             │
│  │  • Mask invalid tokens (e.g., non-ICD-9 codes)       │             │
│  │  • Enforce visit structure (<v> ... <\v>)            │             │
│  │  • Limit sequence length                             │             │
│  │  • Ensure logical code combinations                  │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This document provides visual diagrams for:

1. **System Overview**: 5-phase architecture from data to evaluation
2. **Data Transformation**: MIMIC-III → PatientRecord → Tokens → Batches
3. **Model Architecture**: PromptBartEncoder/Decoder with demographic conditioning
4. **Prompt Mechanism**: How demographics become embeddings and condition generation
5. **Training Flow**: Forward/backward pass, optimizer steps, validation
6. **Vocabulary**: Token ID allocation and 1:1 code mapping
7. **Tensor Shapes**: Complete reference for all stages
8. **Generation**: Autoregressive sampling with demographic conditioning

All diagrams use ASCII/text art for terminal/markdown rendering and include concrete examples with actual tensor shapes and data values.
