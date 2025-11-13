# PromptEHR Onboarding Guide - START HERE

**Last Updated:** November 12, 2025
**Target Audience:** Developers with Python/PyTorch knowledge, zero EHR experience

## 30-Second Elevator Pitch

PromptEHR generates synthetic electronic health records using a fine-tuned BART model. Feed it age/sex demographics, get realistic patient visit histories with ICD-9 diagnosis codes. Think GPT for medical records, but conditioned on patient demographics and trained to avoid medically invalid combinations.

**Key Innovation:** Each medical code = single token (no fragmentation), multi-task learning balances validity with realism, hierarchical generation overcomes sparsity.

## Prerequisites

**Required:**
- Python 3.12+ experience
- PyTorch fundamentals (tensors, nn.Module, training loops)
- Basic understanding of transformers (attention, encoder-decoder)

**Helpful but not required:**
- Medical coding systems (ICD-9/ICD-10)
- MIMIC-III dataset familiarity
- Seq2seq models (BART, T5)

## Complete Learning Path

Total estimated time: **18-24 hours** of focused study across 21 detailed pages.

---

## PHASE 1: Foundation & Problem Space (4-5 hours)

### Understanding the Domain

**[01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md)** (30 min)
- Electronic Health Records basics
- MIMIC-III dataset structure (PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv)
- ICD-9 diagnosis coding system
- Why synthetic data matters (privacy, research, testing)
- Real-world use cases

**[02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)** (45 min)
- Why medical codes ≠ natural language
- Structured vs unstructured data
- ICD-9 format: "401.9", "250.00", "V58.61"
- Subword tokenization problem preview
- Clinical semantics vs linguistic semantics

**[03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md)** (60 min)
- October 9, 2025: Original implementation failure
- How BART tokenizer fragmented codes: "401.9" → ["401", ".", "9"]
- Generation output: Gibberish like "56 ASIAN M <demo> ASIAN m <demoing>"
- Root cause analysis: Random embeddings, no gradient flow
- Why special tokens weren't registered
- **Key Lesson:** Medical codes require 1:1 token mapping
- Files: deprecated/legacy_implementations/main.py

**[04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)** (45 min)
- Evolution from text to token representation
- DiagnosisVocabulary: 1:1 code → ID mapping
- DiagnosisCodeTokenizer: Custom tokenization without fragmentation
- Before/after comparison
- Design principle: Semantic integrity

**[05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md)** (60 min)
- PatientRecord class structure (data_loader.py:40-73)
- Demographics: age, gender (why race was removed)
- Visit structure: List of lists of ICD-9 codes
- Hands-on: Load MIMIC-III and inspect a patient
- Data preprocessing pipeline
- Age calculation, gender encoding (0=M, 1=F)

---

## PHASE 2: Data Pipeline Architecture (4-5 hours)

### Core Data Processing Components

**[06_VOCABULARY_SYSTEM.md](06_VOCABULARY_SYSTEM.md)** (60 min)
- DiagnosisVocabulary class (vocabulary.py)
- Building vocabulary from MIMIC-III
- Code frequency analysis (6,985 unique codes)
- code2idx and idx2code mappings
- Why 1:1 mapping is critical
- Vocabulary statistics: Distribution, long tail

**[07_TOKENIZATION_ARCHITECTURE.md](07_TOKENIZATION_ARCHITECTURE.md)** (75 min)
- DiagnosisCodeTokenizer class (code_tokenizer.py)
- Seven special tokens: <s>, <pad>, </s>, <unk>, <v>, <\v>, <mask>
- Token ID layout: 0-6 (special) + 7+ (diagnosis codes)
- code_offset = 7: Why it matters
- Token sequence format: `<s> <v> code1 code2 <\v> <v> code3 <\v> </s>`
- Encoding/decoding process
- Padding and masking strategies

**[08_DATA_LOADING.md](08_DATA_LOADING.md)** (60 min)
- data_loader.py deep dive
- load_mimic_data() function (data_loader.py:100-250)
- Processing PATIENTS.csv: Age calculation, gender normalization
- Processing ADMISSIONS.csv: Admission dates, demographics
- Processing DIAGNOSES_ICD.csv: Grouping by admission
- Building PatientRecord objects
- Ethnicity normalization (why it was later removed)
- Memory optimization for 46K patients

**[09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)** (90 min)
- EHRPatientDataset class (dataset.py)
- Why data corruption? BART-style denoising
- Corruption functions:
  - Mask infilling (λ=3 Poisson span selection)
  - Token deletion (15% probability)
  - Token replacement (15% probability with random code)
  - Next-visit prediction (for TPL metric)
- EHRDataCollator: Sample expansion
- corrupt_sequence() implementation (dataset.py:200-350)
- Corruption probability = 0.5
- Hands-on: Corrupt a patient sequence and visualize

**[10_DATA_FLOW_INTEGRATION.md](10_DATA_FLOW_INTEGRATION.md)** (45 min)
- End-to-end data pipeline
- MIMIC-III CSVs → PatientRecord → Tokenization → Corruption → Batches
- DataLoader configuration (batch_size=8, num_workers=4)
- Train/validation split (80/20)
- Collate function mechanics
- Label preparation: Setting padding to -100
- Complete example walkthrough

---

## PHASE 3: Model Architecture Deep Dive (5-6 hours)

### Understanding PromptBart Components

**[11_BART_FOUNDATIONS.md](11_BART_FOUNDATIONS.md)** (45 min)
- BART architecture overview (if unfamiliar)
- Encoder-decoder vs decoder-only
- Why we chose BART over GPT
- facebook/bart-base: 6 layers, 768 dim, 16 heads
- BartConfig parameters
- Pretrained weights: What we keep vs reinitialize

**[12_CONDITIONAL_PROMPT.md](12_CONDITIONAL_PROMPT.md)** (75 min)
- ConditionalPrompt class (conditional_prompt.py)
- Demographic conditioning: age (continuous) + gender (categorical)
- Reparameterization: Why d_hidden=128 bottleneck?
- Embedding architecture:
  - Continuous: Linear(n_num, d_hidden) → LayerNorm → Linear(d_hidden, d_model * prompt_length)
  - Categorical: Embedding(cardinality, d_hidden) → Linear(d_hidden, d_model * prompt_length)
- Offset-based categorical embeddings
- Prompt length = 1 (why not more?)
- Forward pass: (x_num, x_cat) → [batch, prompt_length, d_model]

**[13_PROMPT_INJECTION.md](13_PROMPT_INJECTION.md)** (90 min)
- PromptBartEncoder class (prompt_bart_encoder.py)
- PromptBartDecoder class (prompt_bart_decoder.py)
- How prompts are injected at every layer
- Encoder prompt injection: Prepended to input sequence
- Decoder prompt injection: Prepended to decoder sequence
- Attention mask extension for prompts
- Position embedding handling
- Why dual prompts (encoder + decoder)?
- prompt_encoder vs prompt_decoder (separate parameters)

**[14_MAIN_MODEL_ARCHITECTURE.md](14_MAIN_MODEL_ARCHITECTURE.md)** (75 min)
- PromptBartModel class (prompt_bart_model.py:16-150)
- Model initialization: Replacing BART encoder/decoder
- encoder_prompt_encoder vs decoder_prompt_encoder
- Vocabulary resizing (7 special + 6,985 codes = 6,992 tokens)
- Embedding layer extension
- num_prompts calculation
- Forward pass structure
- Input processing flow

**[15_MULTITASK_LEARNING.md](15_MULTITASK_LEARNING.md)** (90 min)
- PromptBartWithDemographicPrediction (prompt_bart_model.py:153-350)
- Why multi-task learning? Medical validity constraints
- Token-level prediction heads:
  - Age prediction: Linear(d_model, 1) per token
  - Sex prediction: Linear(d_model, 2) per token
- Why token-level? Stronger signal than sequence-level
- Loss computation:
  - Language modeling loss (cross-entropy)
  - Age prediction loss (MSE, masked to non-padding)
  - Sex prediction loss (cross-entropy, masked)
  - Combined: `total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss`
- Loss weight evolution: 0.3/0.2 → 0.01/0.2 → 0.001/0.001
- October 23-24, 2025: The validity vs coherence crisis
- Why low weights (0.001)? Prioritize LM loss

**[16_MODEL_INTEGRATION.md](16_MODEL_INTEGRATION.md)** (60 min)
- Complete forward pass walkthrough
- Input preparation: (input_ids, attention_mask, labels, x_num, x_cat)
- Prompt generation: encoder_prompt_encoder(x_num, x_cat)
- Encoder forward: Input embeddings + prompt injection
- Decoder forward: Decoder input + prompt injection + cross-attention
- Loss computation across all heads
- Gradient flow analysis
- Parameter count: 107.5M total
- Memory footprint estimation

---

## PHASE 4: Training System (4-5 hours)

### Training Pipeline Components

**[17_CONFIGURATION.md](17_CONFIGURATION.md)** (45 min)
- config.py structure
- Hyperparameter categories:
  - Data: n_patients, max_seq_len, train_val_split
  - Model: base_model, continuous_features, categorical_features, prompt_length
  - Training: batch_size, epochs, learning_rate, warmup_steps
  - Corruption: lambda_poisson, deletion_prob, replacement_prob, corruption_prob
  - Evaluation: compute_tpl, compute_spl
- Why these specific values?
- Adjusting for experiments

**[18_TRAINING_LOOP.md](18_TRAINING_LOOP.md)** (90 min)
- trainer.py structure (trainer.py:1-500)
- main() function: Setup, data loading, model initialization
- train_epoch() function (trainer.py:100-250):
  - Batch iteration
  - Forward pass
  - Loss computation
  - Backward pass and optimization
  - Metrics tracking
- validate_epoch() function (trainer.py:260-400):
  - Validation batch processing
  - Reconstruction Jaccard calculation
  - No gradient computation
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Learning rate scheduler: Linear warmup (1000 steps)
- Checkpoint saving: Best validation loss

**[19_METRICS_TRACKING.md](19_METRICS_TRACKING.md)** (60 min)
- MetricsTracker class (metrics.py:1-250)
- Tracked metrics:
  - Loss (total combined loss)
  - Perplexity (exp(lm_loss))
  - Token accuracy (all tokens)
  - Code accuracy (diagnosis codes only)
  - Component losses: lm_loss, age_loss, sex_loss, cooccur_loss
  - Reconstruction Jaccard (validation only)
- update() method: Accumulating metrics per batch
- get_average_metrics(): Epoch-level aggregation
- get_last_metrics(): Most recent batch
- reset(): Clearing for new epoch
- Why track multiple metrics? Debugging multi-task learning

**[20_SLURM_TRAINING.md](20_SLURM_TRAINING.md)** (45 min)
- train.slurm structure
- SLURM directives:
  - Job name, partition, resources
  - GPU allocation (1x NVIDIA H200)
  - Time limit (4 hours)
  - Email notifications
- Environment setup: Conda, CUDA
- Job submission: `sbatch train.slurm`
- Monitoring: `squeue -u $USER`, `tail -f logs/train_JOBID.out`
- Common SLURM errors and fixes
- Checkpoint resumption

**[21_LABEL_MASKING.md](21_LABEL_MASKING.md)** (30 min)
- Why set padding tokens to -100?
- PyTorch cross-entropy loss ignores -100
- Prevents model from learning to predict padding
- Implementation in dataset collate function
- Impact on loss computation
- Common bug: Forgetting to mask leads to inflated accuracy

---

## PHASE 5: Generation & Evaluation (3-4 hours)

### Generating Synthetic Patients

**[22_GENERATION_MODES.md](22_GENERATION_MODES.md)** (60 min)
- generate.py structure
- Two generation modes:
  1. Conditional (Reconstruction): Partial code prompt → Full sequence
  2. Zero-prompt (Fully Synthetic): Demographics only → Full sequence
- generate_sequences() function (generate.py:50-150)
- model.generate() parameters:
  - max_length=256
  - num_beams=1 (greedy)
  - do_sample=False
  - no_repeat_ngram_size=1 (prevents duplicate codes)
  - eos_token_id, pad_token_id
- Decoding token IDs back to ICD-9 codes
- Visit boundary detection: <v> and <\v> tokens

**[23_CONDITIONAL_GENERATION.md](23_CONDITIONAL_GENERATION.md)** (45 min)
- Reconstruction evaluation setup
- Prompt creation: First 50% of codes
- Forward generation: Model completes remaining 50%
- Jaccard similarity: Intersection / Union
- Why Jaccard? Measures exact basket match
- Limitations: Doesn't assess clinical plausibility
- User quote (Oct 24): "The goal isn't to predict the exact same basket"
- Reconstruction results: ~0.40-0.45 typical

**[24_ZERO_PROMPT_GENERATION.md](24_ZERO_PROMPT_GENERATION.md)** (60 min)
- Fully synthetic patient generation
- Input: Only age and sex
- Model starts with <s> <v> and generates everything
- Sampling strategies: Greedy vs beam search vs nucleus
- Why greedy for medical codes? Deterministic, no hallucination
- test_unconditional.py: Quick test script
- Output format: PatientRecord objects

**[25_MEDICAL_VALIDITY.md](25_MEDICAL_VALIDITY.md)** (75 min)
- evaluate_medical_validity.py deep dive
- Age-appropriateness rules (docs/reference/medical_validity.md):
  - Pediatric codes (V20.x, V21.x): Age < 18
  - Geriatric codes: Age > 65
  - Pregnancy codes (630-679): Ages 15-50
- Sex-appropriateness rules:
  - Male-only: Prostate (600-608), testicular
  - Female-only: Pregnancy (630-679), ovarian, cervical
- Duplicate detection: Set vs list length
- Metrics calculation: Percentage violations
- Current performance: 99% age, 96% sex, 0% duplicates

**[26_SEMANTIC_COHERENCE.md](26_SEMANTIC_COHERENCE.md)** (90 min)
- evaluate_semantic_coherence.py structure
- Why semantic coherence? Medical validity ≠ statistical plausibility
- Metrics:
  1. **JS Divergence** (Jensen-Shannon): Distribution similarity (target <0.3)
  2. **Top-100 Code Overlap**: Frequency ranking match (target >0.5)
  3. **Co-occurrence Score**: Avg observed pairs per code (target >20)
  4. **Distribution Matching**: KS tests for visits/codes per patient
- Building real distribution from training data
- Building synthetic distribution from generated patients
- October 24 crisis: JS divergence 0.61, top-100 0.04, co-occurrence 3.39
- Root cause: High auxiliary weights (0.01, 0.2) destroyed LM loss dominance

---

## PHASE 6: Advanced Features - Hierarchical Generation (4-5 hours)

### Recent Innovations (November 2025)

**[27_COOCCURRENCE_PROBLEM.md](27_COOCCURRENCE_PROBLEM.md)** (60 min)
- Semantic coherence crisis deep dive
- Why codes are medically valid but statistically implausible
- Flat code vocabulary: 6,985 codes → 48.7M possible pairs
- Observed in training: 696K pairs (1.4% coverage)
- Generated patients: Rare pair combinations
- Example: Hypertension (401.9) appears but never with heart failure (428.0)
- Why LM loss alone insufficient? No explicit pair-level penalty
- Solution preview: Co-occurrence regularization

**[28_COOCCURRENCE_REGULARIZATION.md](28_COOCCURRENCE_REGULARIZATION.md)** (90 min)
- cooccurrence_utils.py deep dive
- Building co-occurrence matrix (sparse CSR format)
- Matrix dimensions: [vocab_size, vocab_size]
- Counting code pairs across training patients
- Co-occurrence threshold: 5 (pairs seen < 5 times are rare)
- cooccurrence_loss_efficient() function (cooccurrence_utils.py:200-300):
  - Extract code indices from generated sequence
  - Create pair indices: cartesian product
  - Lookup co-occurrence counts in matrix
  - Apply threshold: penalize pairs with count < 5
  - Loss = mean penalty across all pairs
- Loss weight: 0.05 (balance with LM loss)
- Integration into training loop

**[29_ICD9_HIERARCHY.md](29_ICD9_HIERARCHY.md)** (75 min)
- ICD-9 code structure: XXX.XX format
- Category extraction: First 3 digits (401.9 → 401)
- icd9_hierarchy.py: ICD9Hierarchy class
- Building hierarchy from vocabulary:
  - 6,985 specific codes → 943 unique categories
  - Average 7.4 codes per category
- Sparsity reduction: 7.4x fewer tokens
- Category-level co-occurrence:
  - 943 categories → 889K possible pairs
  - Observed: 227K pairs (26% coverage)
- Coverage improvement: 1.4% → 26% (18.6x improvement)
- Why this helps? Model learns category patterns, then refines

**[30_HIERARCHICAL_TOKENIZER.md](30_HIERARCHICAL_TOKENIZER.md)** (90 min)
- HierarchicalDiagnosisTokenizer class (hierarchical_tokenizer.py)
- Dual vocabulary structure:
  - Special tokens: 0-6 (same as before)
  - Category tokens: 7-949 (943 categories)
  - Code tokens: 950-7934 (6,985 codes)
- Total vocabulary size: 7,935 tokens
- category_offset = 7, code_offset = 950
- Two-stage encoding:
  1. Categories: [<s>, <v>, cat1, cat2, <\v>, <v>, cat3, <\v>, </s>]
  2. Codes: [<s>, <v>, code1, code2, <\v>, <v>, code3, <\v>, </s>]
- has_vocab attribute vs has_hierarchy (compatibility)
- Tokenizer compatibility fixes (Nov 11, 2025)

**[31_HIERARCHICAL_DATASET.md](31_HIERARCHICAL_DATASET.md)** (60 min)
- HierarchicalEHRDataset class (hierarchical_dataset.py)
- Converts patient codes → categories for training
- Uses ICD9Hierarchy for mapping
- Same corruption functions applied to category sequences
- Why train on categories? Better co-occurrence coverage
- Data collation for hierarchical tokenizer
- Category-level batches

**[32_TWO_STAGE_GENERATION.md](32_TWO_STAGE_GENERATION.md)** (90 min)
- hierarchical_generation.py: Two-stage generation
- Stage 1: Generate category sequence
  - Input: Demographics (age, sex)
  - Output: [<s>, <v>, cat1, cat2, <\v>, <v>, cat3, <\v>, </s>]
- Stage 2: Expand categories → specific codes
  - For each category: Sample codes from hierarchy
  - Maintain visit structure (<v> boundaries)
  - Filter by demographics (age/sex appropriateness)
- generate_hierarchical() function (hierarchical_generation.py:50-200)
- Code sampling strategies:
  - Uniform: Equal probability per code in category
  - Frequency-weighted: Based on training distribution
- Current choice: Frequency-weighted for realism

**[33_HIERARCHICAL_TRAINING.md](33_HIERARCHICAL_TRAINING.md)** (75 min)
- trainer_hierarchical.py: Complete training pipeline
- Differences from trainer.py:
  - Uses HierarchicalDiagnosisTokenizer (7,935 vocab)
  - Uses HierarchicalEHRDataset (category sequences)
  - Uses ICD9Hierarchy for category mapping
  - Integrates co-occurrence loss at code level
- Training on categories but regularizing on codes
- Why? Category-level patterns + code-level realism
- Current training job (Job 5755517): Nov 11-12, 2025
- Expected results: JS divergence <0.3, co-occurrence >20

---

## PHASE 7: Historical Context & Debugging (3-4 hours)

### Learning from the Past

**[34_TIMELINE_OF_CHANGES.md](34_TIMELINE_OF_CHANGES.md)** (60 min)
- October 9, 2025: Fragmentation incident → Token-based approach
- October 12-17, 2025: Phased rebuild (Phases 1-3)
- October 19, 2025: Phase 9 reparameterization (d_hidden=128)
- October 23, 2025: Multi-task learning implementation
- October 24, 2025: Semantic coherence crisis → Reduce weights
- November 11-12, 2025: Hierarchical generation implementation
- Visual timeline with before/after metrics
- Evolution diagram: main.py → Phase 3 → Multi-task → Hierarchical

**[35_KEY_INCIDENTS.md](35_KEY_INCIDENTS.md)** (90 min)
- **Incident 1:** Fragmentation (Oct 9, 2025)
  - Symptom: Gibberish generation
  - Root cause: BART tokenizer subword splitting
  - Fix: DiagnosisCodeTokenizer with 1:1 mapping
  - Files: deprecated/legacy_implementations/main.py
  - Lesson: Medical codes are atomic units

- **Incident 2:** Multi-task weights crisis (Oct 23-24, 2025)
  - Symptom: Perfect medical validity (99%), terrible coherence (JS 0.61)
  - Root cause: Auxiliary weights (0.01, 0.2) dominated LM loss
  - Fix: Reduce to (0.001, 0.001)
  - Files: trainer.py, prompt_bart_model.py
  - Lesson: Multi-task learning requires careful balancing

- **Incident 3:** Tokenizer compatibility (Nov 11, 2025)
  - Symptom: AttributeError: 'HierarchicalDiagnosisTokenizer' object has no attribute 'vocab'
  - Root cause: cooccurrence_utils.py assumed flat tokenizer structure
  - Fix: Add type detection with hasattr() (cooccurrence_utils.py:261-269)
  - Lesson: Abstractions need compatibility layers

- **Incident 4:** MetricsTracker TypeError (Nov 11, 2025)
  - Symptom: TypeError: update() got an unexpected keyword argument 'cooccur_loss'
  - Root cause: trainer_hierarchical.py passes new metric, MetricsTracker doesn't accept it
  - Fix: Extend MetricsTracker.update() signature + tracking (metrics.py:107,119,144,173,203)
  - Lesson: Extending systems requires updating all integration points

**[36_DESIGN_DECISIONS.md](36_DESIGN_DECISIONS.md)** (75 min)
- **Decision 1:** Why remove race from demographics?
  - Medical validity issues: Many codes race-agnostic
  - Bias concerns: Model might learn racial biases from data
  - Solution: Focus on age/sex only (clearer medical validity rules)
  - When: October 23, 2025 (multi-task implementation)

- **Decision 2:** Why reduce auxiliary weights to 0.001?
  - Problem: High weights (0.01, 0.2) destroyed semantic coherence
  - Trade-off: Slight medical validity reduction (99% → ~95%) acceptable
  - Rationale: LM loss must dominate to learn realistic distributions
  - Alternative considered: Hard constraints (rejected - too rigid)

- **Decision 3:** Why BART over GPT?
  - BART: Encoder-decoder (bidirectional encoding)
  - GPT: Decoder-only (causal, more efficient)
  - Choice: BART for PromptEHR paper compatibility + infilling tasks
  - Trade-off: Accept inefficiency for richer context

- **Decision 4:** Why hierarchical generation?
  - Problem: Flat codes have 1.4% co-occurrence coverage
  - Solution: Category-level training (26% coverage, 18.6x improvement)
  - Cost: Two-stage generation complexity
  - Benefit: Dramatically improved semantic coherence

**[37_DEPRECATED_CODE.md](37_DEPRECATED_CODE.md)** (45 min)
- deprecated/ directory structure
- legacy_implementations/main.py: Original text-based implementation
- backups/v2/: Pre-reparameterization code
- backups/v3/: Pre-multi-task code
- unit_tests/: Phase 1-3 validation tests (test_phase1.py, test_phase2.py, test_phase3.py)
- utilities/: One-off scripts (analyze_generated.py, decode_patients.py)
- Why deprecated? Superseded by evolved architecture
- When to reference? Understanding evolution, comparing approaches

**[38_DEBUGGING_GUIDE.md](38_DEBUGGING_GUIDE.md)** (60 min)
- Common errors and solutions:
  - **RuntimeError: CUDA out of memory** → Reduce batch_size
  - **ValueError: Target size mismatch** → Check label masking (-100 for padding)
  - **AttributeError on tokenizer** → Check flat vs hierarchical tokenizer
  - **Loss is NaN** → Check learning rate, gradient clipping
  - **Perplexity exploding** → Vocabulary mismatch, check embedding initialization
- Debugging workflow:
  1. Check logs: logs/train_JOBID.err
  2. Inspect data: Print batch shapes, token IDs
  3. Validate preprocessing: Check corruption output
  4. Check model: Print parameter counts, gradients
- Tools: pdb, torch.autograd.set_detect_anomaly(True)
- Slurm-specific debugging: Interactive sessions, job arrays

---

## PHASE 8: Reference & Quick Tasks (1-2 hours)

### Day-to-Day Operations

**[39_FILE_REFERENCE.md](39_FILE_REFERENCE.md)** (45 min)
- Complete file → purpose mapping (all 29 Python files)
- Grouped by category:
  - Data: data_loader.py, vocabulary.py, code_tokenizer.py, dataset.py
  - Model: prompt_bart_model.py, prompt_bart_encoder.py, prompt_bart_decoder.py, conditional_prompt.py
  - Training: trainer.py, trainer_hierarchical.py, config.py, metrics.py
  - Generation: generate.py, hierarchical_generation.py
  - Evaluation: evaluate_medical_validity.py, evaluate_semantic_coherence.py
  - Hierarchy: icd9_hierarchy.py, hierarchical_tokenizer.py, hierarchical_dataset.py
  - Utils: cooccurrence_utils.py
  - Tests: test_*.py (12 files)
- Import dependencies graph
- Where to find specific functionality

**[40_COMMON_TASKS.md](40_COMMON_TASKS.md)** (60 min)
- **Task 1:** Train a model
  ```bash
  # Edit config.py for hyperparameters
  sbatch train.slurm  # Flat training
  sbatch train_hierarchical.slurm  # Hierarchical training
  ```

- **Task 2:** Generate synthetic patients
  ```bash
  python test_unconditional.py  # Quick test (10 patients)
  python evaluate_medical_validity.py  # 100 patients with validity check
  ```

- **Task 3:** Evaluate a trained model
  ```bash
  python evaluate_semantic_coherence.py --checkpoint checkpoints/best_model.pt
  python evaluate_medical_validity.py --checkpoint checkpoints/best_model.pt
  ```

- **Task 4:** Inspect a patient
  ```python
  from data_loader import load_mimic_data
  patients = load_mimic_data(n_patients=100)
  patient = patients[0]
  print(f"Age: {patient.age}, Gender: {patient.gender}")
  for i, visit in enumerate(patient.visits):
      print(f"Visit {i+1}: {visit}")
  ```

- **Task 5:** Debug tokenization
  ```python
  from vocabulary import DiagnosisVocabulary
  from code_tokenizer import DiagnosisCodeTokenizer
  vocab = DiagnosisVocabulary(codes=["401.9", "250.00"])
  tokenizer = DiagnosisCodeTokenizer(vocab)
  codes = ["401.9", "250.00"]
  token_ids = tokenizer.encode_codes(codes)
  decoded = tokenizer.decode_codes(token_ids)
  print(f"Original: {codes}")
  print(f"Token IDs: {token_ids}")
  print(f"Decoded: {decoded}")
  ```

- **Task 6:** Monitor training
  ```bash
  squeue -u $USER  # Check job status
  tail -f logs/train_JOBID.out  # Real-time log viewing
  grep "Validation" logs/train_JOBID.out  # Extract validation metrics
  ```

**[41_CONFIGURATION_REFERENCE.md](41_CONFIGURATION_REFERENCE.md)** (30 min)
- config.py complete breakdown
- Every parameter explained with:
  - Purpose
  - Typical values
  - Impact of changes
  - Experimentation guidance
- Examples:
  - n_patients: 25000 → 50000 (more data, slower training)
  - learning_rate: 1e-4 → 5e-5 (fine-tuning, slower convergence)
  - age_loss_weight: 0.001 → 0.01 (more medical validity, less coherence)
  - cooccurrence_loss_weight: 0.05 → 0.1 (stronger pair-level penalty)

---

## Navigation Guide

### For First-Time Learners
**Recommended Path:** Pages 1-41 in order (18-24 hours total)

**Fast Track (8-10 hours):**
- Pages 1-5 (Foundation)
- Pages 11-16 (Model Architecture)
- Pages 22-26 (Generation & Evaluation)
- Pages 39-41 (Reference)

### For Experienced ML Engineers
**Recommended Path:**
- Skim: Pages 1-2 (Medical context)
- Focus: Pages 6-16 (Architecture), 27-33 (Advanced features)
- Reference: Pages 39-41 (Quick tasks)

### For Medical Domain Experts
**Recommended Path:**
- Skim: Pages 11 (BART basics)
- Focus: Pages 1-5 (EHR context), 15 (Multi-task), 25-26 (Evaluation), 36 (Design decisions)
- Reference: Pages 39-41

### For Debugging Specific Issues
- **Data issues:** Pages 6-10
- **Training issues:** Pages 17-21, 38
- **Generation issues:** Pages 22-24, 32
- **Evaluation issues:** Pages 25-26
- **Historical context:** Pages 34-37

## Current System Status (as of Nov 12, 2025)

**Training:** Hierarchical model Job 5755517 in progress (~3 hours remaining)
**Expected Results:** JS divergence <0.3, co-occurrence >20 (18.6x coverage improvement)
**Branch:** feature/cooccurrence-learning (11 commits)

## Next Steps

**Begin your journey:** [01_WHAT_IS_EHR.md →](01_WHAT_IS_EHR.md)

---

**Pro Tip:** This guide is designed for side-by-side reading with code open. Use file:line references to jump directly to relevant implementations. Each page includes hands-on exercises - do them!
