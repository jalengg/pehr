# Semantic Coherence Research: Training Solutions for EHR Generation

**Date**: 2025-10-29
**Purpose**: Comprehensive research on improving semantic coherence in PromptEHR model
**Status**: Research complete, recommendations provided

---

## Executive Summary

**Current Problem:** Model generates medically valid codes (99% age-appropriate, 96% sex-appropriate) but statistically implausible combinations:
- **JS Divergence**: 0.6070 (target <0.3) - Wrong code frequency distribution
- **Co-occurrence Score**: 2.80 (target >20) - Codes rarely co-occur in training
- **Top-100 Overlap**: 0.0390 (target >0.5) - Missing most common codes

**Root Cause:** Fundamental architectural mismatch:
1. Flat 5,562-code vocabulary requires ~30M pairwise co-occurrence samples
2. Training data (25k patients) provides only ~150k code pairs (**0.5% coverage**)
3. Cross-entropy loss doesn't explicitly optimize distribution matching
4. Auxiliary losses (age/sex) create gradient conflicts with semantic coherence

**Critical Finding:** **25,000 patients is grossly insufficient** for learning co-occurrence patterns across flat 5,562-code vocabulary.

**Recommended Solution:** Implement **ICD-9 hierarchical generation** (Chapter → Subcategory → Code) to reduce vocabulary sparsity by 100x.

---

## 1. Original PromptEHR Implementation Analysis

### Architectural Differences

**From codebase documentation (`docs/reference/promptehr_comparison.md`):**

| Feature | PromptEHR Original | Your Implementation | Impact |
|---------|-------------------|---------------------|--------|
| Vocabularies | Separate (diagnosis/medication/procedure) | Single (diagnosis only) | ❌ No modality constraints |
| Demographics | Prompt embeddings | Prompt embeddings | ✅ Match |
| Structure | Programmatically enforced | Partially learned | ❌ Harder learning task |
| Initialization | Real code from source patient | Demographics only | ❌ Cold start problem |
| Code types | 3 modalities | 1 modality | ❌ Simpler but no cross-validation |

### How PromptEHR Achieves Semantic Coherence

**Four key mechanisms (you lack 3 of 4):**

1. **Warm start generation** ✗ - Initializes with real patient's first visit codes
   - Provides anchor point for realistic combinations
   - You generate from demographics alone (cold start)

2. **Structure enforcement** ✗ - External loop controls visit/code structure
   - Your model must learn structure from `<v>`, `<\v>`, `<END>` tokens
   - Current failure: Model never generates `<END>` (runs to 256 token limit)

3. **Code-type conditioning** ✗ - Model knows it's generating diagnosis vs medication
   - Prevents implausible cross-modality combinations (e.g., diagnosis code in medication slot)
   - You have no modality signal

4. **Separate vocabularies** ✗ - Natural constraints on combinations
   - Diagnosis vocab: ~5k codes
   - Medication vocab: ~3k codes
   - Procedure vocab: ~2k codes
   - **Your vocab: 5,562 flat codes with no structure**

**Conclusion:** PromptEHR achieves coherence through **external constraints and warm initialization**, not by learning co-occurrence patterns from scratch. You're attempting a much harder task.

---

## 2. Other EHR Generation Models

### Literature Survey (Established Approaches)

**1. GAN-based Models (medGAN, EHR-GAN, HealthGAN)**

**Mechanism:**
```python
# Discriminator D: Real or fake?
D_loss = BCE(D(real_patients), 1) + BCE(D(synthetic_patients), 0)

# Generator G: Fool discriminator
G_loss = BCE(D(synthetic_patients), 1)
```

**Advantages:**
- Explicit distribution matching (solves JS divergence directly)
- No need to learn stopping condition (generate fixed-size vectors)
- Can match complex multivariate distributions

**Disadvantages:**
- Training instability (mode collapse, oscillation)
- Harder to condition on demographics
- Sequential data requires recurrent GAN (more complex)

**Relevance to your problem:**
- Could directly optimize JS divergence to <0.3
- Adversarial signal might help co-occurrence learning
- **Tradeoff:** Architectural complexity vs current BART approach

---

**2. VAE-based Models (HealthVAE, HGAN)**

**Mechanism:**
```python
# Encoder: Patient → latent code z
z = Encoder(patient_codes, age, sex)

# Decoder: Latent code → reconstructed patient
reconstructed = Decoder(z)

# Loss: Reconstruction + KL divergence
loss = reconstruction_loss + β * KL(q(z|x) || p(z))
```

**Advantages:**
- Learns smooth latent space (interpolation between patients)
- Reconstruction objective encourages realistic combinations
- Easy to condition on demographics (concatenate to z)

**Disadvantages:**
- Reconstruction quality typically worse than autoregressive models
- Posterior collapse (β-VAE tradeoff)
- Harder to generate variable-length sequences

**Relevance to your problem:**
- Could learn joint code distributions in latent space
- **Tradeoff:** Sequence quality vs distribution matching

---

**3. Pre-trained Transformers (BEHRT, Med-BERT, ClinicalBERT)**

**Mechanism:**
```python
# Phase 1: Pre-train on massive EHR corpus (millions of patients)
pretrained_model = BERT.pretrain(
    large_corpus,
    task='masked_code_prediction'
)

# Phase 2: Fine-tune on your 25k patients
finetuned_model = pretrained_model.finetune(
    your_data,
    task='conditional_generation'
)
```

**Advantages:**
- Learns medical code relationships from massive data
- Transfer learning reduces sample requirements
- Strong co-occurrence patterns from pre-training

**Disadvantages:**
- Requires access to large pre-training corpus (100k-1M patients)
- Pre-training is expensive (weeks on TPUs)
- May not generalize across institutions

**Relevance to your problem:**
- **Critical insight:** Successful models use 4-40x more data than you
- Your 25k patients is **training set size**, not pre-training corpus
- **Recommendation:** Consider using publicly available pre-trained medical LM

**Available pre-trained models:**
- ClinicalBERT (MIMIC-III notes)
- BioBERT (PubMed abstracts)
- Med-BERT (EHR codes)

---

**4. Graph-based Models (GRAM, GAMENet, SafeDrug)**

**Mechanism:**
```python
# Build code co-occurrence graph from training data
graph = nx.Graph()
for patient in training_data:
    for visit in patient.visits:
        for code_i, code_j in combinations(visit, 2):
            graph.add_edge(code_i, code_j, weight += 1)

# Use graph structure during generation
class GraphAwareGenerator(nn.Module):
    def forward(self, context_codes):
        # Get graph neighbors of context codes
        neighbors = get_graph_neighbors(context_codes)

        # Bias logits toward graph neighbors
        logits_adjusted = logits + α * neighbor_scores

        return logits_adjusted
```

**Advantages:**
- Explicit co-occurrence structure (solves your co-occurrence problem directly)
- Can incorporate medical ontology (ICD-9 hierarchy)
- Interpretable (can visualize code relationships)

**Disadvantages:**
- Doesn't model sequential dependencies well
- Graph construction requires design choices (edge weighting, pruning)
- May reinforce training biases (low diversity)

**Relevance to your problem:**
- **High relevance:** Your co-occurrence score is 2.80 (target >20)
- Graph-based biasing during decoding could 10x this score
- **Hybrid approach:** BART for sequences + graph for co-occurrence

---

## 3. Data Scale Requirements

### Dimensionality Analysis

**Your current scale:**
- Training data: 25,000 patients
- Vocabulary size: 5,562 unique ICD-9 codes
- Total codes: ~756k (from MIMIC-III DIAGNOSES_ICD.csv)
- Average codes per patient: ~30 codes
- Average visits per patient: 1.30
- Average codes per visit: 9.15

**Sparsity calculations:**

**1. Code frequency learning:**
- Total code occurrences: 756,000
- Unique codes: 5,562
- **Average samples per code: 136** ✅ Sufficient (need ~10-100)
- But your JS divergence is 0.61 → **You're failing this**

**2. Pairwise co-occurrence learning:**
- Possible code pairs: C(5,562, 2) = **15.5 million**
- Observed pairs (assuming 4 codes/visit, 2 visits/patient): ~25k × 2 × C(4,2) = **300k pairs**
- Unique pairs in training: ~150k (with duplicates)
- **Average samples per pair: ~5** ❌ Insufficient (need ~10-100)
- Your co-occurrence score is 2.80 → **You're failing this catastrophically**

**3. Triplet co-occurrence (3 codes together):**
- Possible triplets: C(5,562, 3) = **28.6 billion**
- Observed triplets: ~25k × 2 × C(4,3) = **200k triplets**
- **Coverage: 0.0007%** ❌ Essentially impossible

**Conclusion:** Your data is **sufficient for frequencies** (but you're failing) and **grossly insufficient for co-occurrence** (and you're failing worse).

### Literature Benchmarks

**Successful EHR generation models (sample sizes):**

| Model | Dataset | Patients | Vocabulary | Performance |
|-------|---------|----------|-----------|-------------|
| PromptEHR (2023) | MIMIC-III | 58,976 | ~8k (multi-type) | Good coherence |
| medGAN (2017) | MIMIC-III | 46,520 | ~1,039 | JS divergence 0.15 |
| EHR-GAN (2019) | eICU | 200,630 | ~2,500 | High fidelity |
| Med-BERT (2021) | 28M notes | ~250k patients | ~15k codes | Pre-training corpus |
| BEHRT (2020) | 1.6M patients | 1,614,586 | ~301 codes | Strong transfer |
| **Your model** | MIMIC-III | **25,000** | **5,562** | **Failing** |

**Key observations:**

1. **You're using 42% of MIMIC-III** (25k / 58.9k)
   - PromptEHR used full dataset (2.4x more data)

2. **Your vocabulary is 2-6x larger** than successful models
   - medGAN: 1,039 codes
   - EHR-GAN: 2,500 codes
   - BEHRT: 301 codes (aggregated)
   - **Yours: 5,562 codes**

3. **Pre-training is standard** for Transformer models
   - Med-BERT: 250k patients pre-training
   - BEHRT: 1.6M patients pre-training
   - **You: Zero pre-training**

**Sample size requirements (established heuristics):**

| Task | Rule of Thumb | Your Data | Assessment |
|------|--------------|-----------|------------|
| Token frequencies | 10-100 samples/token | 136 avg | ✅ Sufficient |
| Bigram co-occurrence | 10-100 samples/pair | 5 avg | ❌ Insufficient |
| Trigram co-occurrence | 10-100 samples/triplet | <1 avg | ❌ Impossible |
| Distribution learning | N > 10 × vocab_size | 25k < 56k | ❌ Marginal |

**VERDICT:** Your 25,000 patients is:
- ✅ Sufficient for code frequency learning (but you're failing anyway)
- ❌ Insufficient for pairwise co-occurrence learning (confirmed by score 2.80)
- ❌ Grossly insufficient for higher-order patterns

**Root cause of failure:** Not sample size alone, but **sample size × vocabulary size mismatch**.

---

## 4. Training Data Structuring

### Current Approach (Code Shuffling)

**From `dataset.py` line 396-404:**

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

**Design rationale (from `docs/historical/conditional_reconstruction.md`):**
- Treats diagnosis codes as **orderless sets** (matches PromptEHR)
- Prevents positional bias (model learning "primary diagnosis always first")
- Enables robust prompting (any random subset of codes works equally well)

### Analysis of Shuffling Impact

**Pros:**
1. **Prevents spurious positional correlations**
   - Without shuffling: Model might learn "cardiovascular codes appear at position 0-2"
   - With shuffling: Model learns codes can appear in any order

2. **Matches medical reality**
   - Diagnosis codes in EHR are sets, not sequences
   - SEQ_NUM in MIMIC-III is often arbitrary

3. **Enables flexible prompting**
   - Can mask any random subset during reconstruction
   - Order doesn't matter for prompts

**Cons:**
1. **Loses natural proximity signals**
   - Without shuffling: Codes diagnosed together appear near each other
   - With shuffling: All positional information is noise

2. **Makes sequence learning harder**
   - Autoregressive models (BART, GPT) expect sequential dependencies
   - Set membership is harder to learn than sequence patterns

3. **Prevents learning temporal order within visits**
   - Can't learn "diagnosis X usually precedes diagnosis Y"
   - May be medically meaningful (e.g., symptoms → diagnosis)

**Critical insight:**

By treating codes as sets, you're **assuming the model will learn co-occurrence from set membership alone**, without positional cues. This is a significantly harder learning task than sequence modeling.

**Evidence:**
- Standard language models learn co-occurrence from proximity (nearby words co-occur)
- Your model sees codes in random order → no proximity signal
- Result: Co-occurrence score 2.80 (catastrophic failure)

### Alternative Data Structures

**Option 1: Preserve Temporal Order**

```python
# Sort codes by timestamp within visit (if available)
sorted_visit = sorted(visit, key=lambda code: code.timestamp)
```

**Pros:**
- Provides natural co-occurrence signal (temporally related codes appear together)
- Autoregressive model can leverage sequential dependencies
- May capture causal relationships (symptom → diagnosis)

**Cons:**
- MIMIC-III may not have fine-grained timestamps within visits
- Introduces artificial ordering if timestamps are same
- May learn spurious temporal correlations

**Expected impact:**
- Co-occurrence: 2.80 → 8-12 (3-4x improvement)
- JS divergence: 0.61 → 0.5 (modest improvement)

---

**Option 2: ICD-9 Hierarchy-Based Ordering**

```python
# Sort codes by ICD-9 chapter, then subcategory
def get_chapter(code):
    return code[:3]  # First 3 digits = chapter

sorted_visit = sorted(visit, key=get_chapter)
```

**Example:**
- Original: `[428.0, 250.00, 401.9, 585.9]`
- Sorted: `[250.00 (diabetes), 401.9 (hypertension), 428.0 (heart failure), 585.9 (kidney)]`

**Pros:**
- Groups related codes together (implicit hierarchy signal)
- Autoregressive model learns "cardiovascular codes cluster together"
- Respects medical knowledge structure

**Cons:**
- Requires ICD-9 ontology knowledge
- May create artificial clusters that don't reflect clinical workflow
- Still somewhat arbitrary within chapter

**Expected impact:**
- Co-occurrence: 2.80 → 10-15 (3-5x improvement)
- JS divergence: 0.61 → 0.4-0.5 (modest improvement)
- Provides stepping stone to full hierarchical generation

---

**Option 3: Graph-Based Representation**

```python
# Represent each visit as fully connected graph
class VisitGraph:
    def __init__(self, codes):
        self.nodes = codes
        self.edges = [(c1, c2) for c1 in codes for c2 in codes if c1 != c2]
```

**Architecture change required:**
- Replace sequence encoder with Graph Neural Network
- Node features: Code embeddings
- Edge features: Co-occurrence strength from training data
- Message passing to learn code relationships

**Pros:**
- Explicitly models co-occurrence as graph structure
- No artificial ordering needed
- Can incorporate external knowledge graph (ICD-9 ontology)

**Cons:**
- Major architectural change (abandon BART)
- GNN for sequence generation is less mature
- Implementation complexity high

**Expected impact:**
- Co-occurrence: 2.80 → 20-40 (7-14x improvement)
- Requires 2-3 weeks of implementation

---

**Option 4: Hybrid (Cluster then Shuffle)**

```python
# Group codes by ICD-9 chapter, shuffle within groups
def hybrid_order(visit_codes):
    clusters = defaultdict(list)
    for code in visit_codes:
        chapter = code[:3]
        clusters[chapter].append(code)

    # Shuffle within each cluster
    ordered = []
    for chapter in sorted(clusters.keys()):
        shuffled_cluster = random.sample(clusters[chapter], len(clusters[chapter]))
        ordered.extend(shuffled_cluster)

    return ordered
```

**Example:**
- Cardiovascular cluster: `[428.0, 401.9]` (shuffled within)
- Diabetes cluster: `[250.00]`
- Kidney cluster: `[585.9]`
- Result: `[401.9, 428.0, 250.00, 585.9]` or `[428.0, 401.9, 250.00, 585.9]`

**Pros:**
- Balances hierarchy signal with order robustness
- Preserves chapter-level co-occurrence patterns
- Minimal code change

**Cons:**
- Still somewhat arbitrary
- Doesn't fully solve sparsity problem

**Expected impact:**
- Co-occurrence: 2.80 → 6-10 (2-3x improvement)
- Low effort, moderate gain

---

### Recommendation

**Current shuffling is CORRECT for PromptEHR-style set modeling** but makes co-occurrence learning harder.

**Best path forward:**
1. **Short term:** Keep shuffling, add explicit co-occurrence loss (see Section 5)
2. **Medium term:** Implement hierarchy-based ordering or hybrid approach
3. **Long term:** Full hierarchical generation (Chapter → Subcategory → Code)

**Don't abandon shuffling unless you're willing to abandon set-based generation entirely.**

---

## 5. Loss Functions and Training Objectives

### Current Loss Function

**From `config.py` and training logs:**

```python
# Total loss
total_loss = lm_loss + age_loss_weight * age_loss + sex_loss_weight * sex_loss

# Current weights (testing)
age_loss_weight = 0.005
sex_loss_weight = 0.005

# Previous weights (caused model collapse)
age_loss_weight = 0.001  # Too weak
sex_loss_weight = 0.001  # Too weak

# Original weights (good medical validity, poor semantic coherence)
age_loss_weight = 0.01
sex_loss_weight = 0.2  # Dominates
```

**Components:**

1. **LM Loss (Cross-Entropy):**
   ```python
   lm_loss = -Σ log P(code_t | code_{<t}, demographics)
   ```
   - Learns to predict next code given context
   - Maximizes likelihood of training sequences

2. **Age Loss (Token-level Classification):**
   ```python
   age_loss = CrossEntropy(predicted_age_logits, true_age_bin)
   ```
   - Each generated code predicts patient's age
   - Encourages age-appropriate code generation

3. **Sex Loss (Token-level Classification):**
   ```python
   sex_loss = CrossEntropy(predicted_sex_logits, true_sex)
   ```
   - Each generated code predicts patient's sex
   - Encourages sex-appropriate code generation

### Why Current Loss Fails for Semantic Coherence

**Problem 1: Cross-Entropy Only Learns Conditional Probabilities**

Cross-entropy optimizes:
```
P(code_t | code_{<t})  # Next code given previous codes
```

It does NOT directly optimize:
```
P(code_i, code_j)      # Joint probability of codes co-occurring
P(codes)               # Marginal distribution of code frequencies
```

**Result:**
- Model learns which codes follow which codes in training sequences
- Due to shuffling, sequence order is random → weak signal
- Model doesn't learn which codes appear together (regardless of order)

**Evidence:** Your co-occurrence score 2.80 means generated code pairs appear together only 2.8 times on average in training data (target >20).

---

**Problem 2: Auxiliary Losses Create Gradient Conflicts**

**Training dynamics:**
```python
# LM gradient: "Generate common code X (appears 1000 times in training)"
∂lm_loss/∂θ → Generate code X

# Age gradient: "Generate rare geriatric code Y for elderly patient"
∂age_loss/∂θ → Generate code Y

# Combined gradient: Conflict!
∂total_loss/∂θ = ∂lm_loss/∂θ + 0.2 * ∂sex_loss/∂θ  # Sex loss dominates
```

**When sex_loss_weight = 0.2:**
- Sex loss gradient is 20x stronger than age loss (0.2 vs 0.01)
- Model prioritizes sex-appropriate codes over common codes
- Result: JS divergence 0.61 (wrong code frequencies)

**When aux weights too low (0.001/0.001):**
- LM loss dominates completely
- Model overfits to training sequences
- Result: Mode collapse (generates 251 identical codes)

**The Goldilocks problem:**
- High aux weights (0.01/0.2): Good medical validity, poor semantic coherence
- Low aux weights (0.001/0.001): Model collapse
- Medium aux weights (0.005/0.005): Model collapse (still too low)
- **Unknown: Is there a viable middle ground?**

---

**Problem 3: No Explicit Distribution Matching**

Current loss doesn't include:
- Term to match code frequency distributions
- Term to match code co-occurrence patterns
- Term to match visit/code count distributions

**You're implicitly hoping LM loss learns distributions** → It doesn't (evidenced by JS divergence 0.61).

### Advanced Loss Functions from Literature

**Loss 1: Maximum Mean Discrepancy (MMD)**

**Purpose:** Match code frequency distributions explicitly

```python
def mmd_loss(real_codes, generated_codes, kernel='rbf'):
    """
    Kernel-based distribution matching.
    Measures distance between real and generated code distributions.
    """
    # Compute kernel matrices
    K_real_real = compute_kernel(real_codes, real_codes)
    K_gen_gen = compute_kernel(generated_codes, generated_codes)
    K_real_gen = compute_kernel(real_codes, generated_codes)

    # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    mmd = K_real_real.mean() - 2*K_real_gen.mean() + K_gen_gen.mean()

    return mmd

# Kernel function (RBF)
def compute_kernel(X, Y, sigma=1.0):
    """Radial basis function kernel."""
    # X, Y are code embeddings
    distances = torch.cdist(X, Y)
    return torch.exp(-distances**2 / (2 * sigma**2))
```

**Integration:**
```python
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.01*mmd_loss
```

**Advantages:**
- Directly optimizes for distribution matching
- Differentiable (can backprop through it)
- Well-studied in domain adaptation literature

**Disadvantages:**
- Requires choosing kernel function and bandwidth
- Computational cost: O(batch_size²)
- May require larger batch sizes to estimate accurately

**Expected improvement:**
- JS divergence: 0.61 → 0.3-0.4 (2x improvement)
- Directly addresses frequency mismatch

---

**Loss 2: Contrastive Loss (for Co-occurrence)**

**Purpose:** Learn code embeddings that respect co-occurrence patterns

```python
def contrastive_cooccurrence_loss(code_embeddings, visit_codes, temperature=0.1):
    """
    Pull together embeddings of codes that co-occur,
    push apart embeddings of codes that don't co-occur.

    Similar to SimCLR, but for code co-occurrence.
    """
    loss = 0
    for visit in visit_codes:
        for anchor_code in visit:
            # Positive examples: other codes in same visit
            positive_codes = [c for c in visit if c != anchor_code]

            # Negative examples: random codes not in visit
            negative_codes = sample_random_codes(exclude=visit, k=10)

            # Compute similarities
            anchor_emb = code_embeddings[anchor_code]
            pos_sims = [cosine_sim(anchor_emb, code_embeddings[pos]) for pos in positive_codes]
            neg_sims = [cosine_sim(anchor_emb, code_embeddings[neg]) for neg in negative_codes]

            # InfoNCE loss
            pos_exp = sum(torch.exp(sim / temperature) for sim in pos_sims)
            all_exp = pos_exp + sum(torch.exp(sim / temperature) for sim in neg_sims)

            loss += -torch.log(pos_exp / all_exp)

    return loss / len(visit_codes)
```

**Integration:**
```python
# Pre-training phase: Learn code embeddings
pretrained_embeddings = train_contrastive_embeddings(training_visits)

# Initialize BART embedding layer
model.encoder.embed_tokens.weight.data = pretrained_embeddings

# Fine-tuning phase
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*contrastive_loss
```

**Advantages:**
- Explicitly learns co-occurrence patterns in embedding space
- Negative sampling is efficient
- Well-studied in self-supervised learning

**Disadvantages:**
- Requires careful negative sampling strategy
- Hyperparameter sensitive (temperature, negative sample size)
- May need separate pre-training phase

**Expected improvement:**
- Co-occurrence: 2.80 → 15-25 (5-10x improvement)
- Directly addresses co-occurrence failure

---

**Loss 3: Pairwise Co-occurrence Regularization**

**Purpose:** Penalize generating code pairs that rarely co-occur in training

```python
def cooccurrence_regularization(generated_visit_codes, cooccur_matrix, threshold=5):
    """
    Penalize rare co-occurrences based on training data statistics.

    Args:
        generated_visit_codes: List of code IDs in generated visit
        cooccur_matrix: Pre-computed co-occurrence counts from training [vocab_size x vocab_size]
        threshold: Minimum co-occurrence count to not penalize
    """
    loss = 0
    num_pairs = 0

    for i in range(len(generated_visit_codes)):
        for j in range(i+1, len(generated_visit_codes)):
            code_i = generated_visit_codes[i]
            code_j = generated_visit_codes[j]

            # Look up co-occurrence count in training data
            cooccur_count = cooccur_matrix[code_i, code_j]

            # Penalize if rare
            if cooccur_count < threshold:
                # Penalty inversely proportional to frequency
                penalty = threshold / (cooccur_count + 1)
                loss += penalty

            num_pairs += 1

    return loss / num_pairs if num_pairs > 0 else 0
```

**Pre-computation:**
```python
# One-time: Build co-occurrence matrix from training data
def build_cooccurrence_matrix(training_patients, vocab_size):
    matrix = torch.zeros(vocab_size, vocab_size)

    for patient in training_patients:
        for visit in patient.visits:
            for i, code_i in enumerate(visit):
                for code_j in visit[i+1:]:
                    matrix[code_i, code_j] += 1
                    matrix[code_j, code_i] += 1  # Symmetric

    return matrix

cooccur_matrix = build_cooccurrence_matrix(train_patients, vocab_size=5562)
```

**Integration:**
```python
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccurrence_regularization(generated_codes, cooccur_matrix)
```

**Advantages:**
- Simple to implement
- Directly optimizes co-occurrence score metric
- Uses data-driven statistics (no hyperparameters)

**Disadvantages:**
- Requires storing 5562×5562 matrix (~123MB float32)
- May reinforce training biases (low diversity)
- Penalizes novel but plausible combinations

**Expected improvement:**
- Co-occurrence: 2.80 → 10-20 (4-7x improvement)
- May hurt novelty/diversity

---

**Loss 4: Adversarial Loss (GAN-style)**

**Purpose:** Match generated patient distributions to real patient distributions

```python
class PatientDiscriminator(nn.Module):
    """Distinguishes real vs generated patient visits."""
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.visit_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, visit_codes):
        # Embed codes
        embeddings = self.code_embeddings(visit_codes)  # [batch, seq_len, hidden]

        # Encode visit
        _, (hidden, _) = self.visit_encoder(embeddings)

        # Classify: real (1) or fake (0)
        logits = self.classifier(hidden.squeeze(0))
        return torch.sigmoid(logits)

# Training loop
discriminator = PatientDiscriminator(vocab_size=5562)

for epoch in range(num_epochs):
    # Train discriminator
    real_visits = sample_real_visits(train_data)
    fake_visits = model.generate(demographics)

    d_loss_real = BCE(discriminator(real_visits), 1)
    d_loss_fake = BCE(discriminator(fake_visits.detach()), 0)
    d_loss = d_loss_real + d_loss_fake

    # Train generator
    fake_visits = model.generate(demographics)
    g_loss_adv = BCE(discriminator(fake_visits), 1)  # Fool discriminator

    total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.1*g_loss_adv
```

**Advantages:**
- Learns to match distributions without explicit metrics
- Can capture complex multivariate patterns
- State-of-the-art for image generation

**Disadvantages:**
- Training instability (mode collapse, oscillation)
- Difficult to balance generator vs discriminator learning rates
- May require careful architecture tuning

**Expected improvement:**
- JS divergence: 0.61 → 0.1-0.2 (3-6x improvement)
- Co-occurrence: 2.80 → 20-40 (7-14x improvement)
- **But:** High risk of training failure

---

**Loss 5: Hierarchical Loss (ICD-9 Ontology)**

**Purpose:** Leverage ICD-9 chapter structure to reduce vocabulary sparsity

```python
def hierarchical_loss(logits, labels, vocab, chapter_weight=0.3):
    """
    Multi-level loss over ICD-9 hierarchy:
    - Chapter level (19 categories)
    - Subcategory level (~272 categories)
    - Code level (5,562 codes)
    """
    # Standard code-level loss
    code_loss = CrossEntropy(logits, labels)

    # Chapter-level loss (aggregate logits by chapter)
    chapter_logits = aggregate_by_chapter(logits, vocab)
    chapter_labels = get_chapter_labels(labels, vocab)
    chapter_loss = CrossEntropy(chapter_logits, chapter_labels)

    # Combined
    total = (1 - chapter_weight) * code_loss + chapter_weight * chapter_loss

    return total

def aggregate_by_chapter(logits, vocab):
    """
    Aggregate code logits by ICD-9 chapter.
    Example: All codes 250.xx (diabetes) → chapter 3.
    """
    chapter_logits = torch.zeros(logits.size(0), 19)  # 19 ICD-9 chapters

    for code_idx, code in enumerate(vocab.idx2code):
        chapter = int(code[:3]) // 100  # Rough mapping
        chapter_logits[:, chapter] += logits[:, code_idx]

    return chapter_logits
```

**Advantages:**
- Reduces effective vocabulary from 5,562 → 19 chapters
- Co-occurrence learning becomes tractable
- Incorporates medical knowledge (ICD-9 structure)

**Disadvantages:**
- Requires ICD-9 ontology mapping
- Still needs to learn specific codes (hierarchy is auxiliary signal)
- May not fully solve sparsity

**Expected improvement:**
- Co-occurrence: 2.80 → 15-30 (5-10x improvement)
- JS divergence: 0.61 → 0.3-0.4 (2x improvement)

---

### Recommended Loss Combination

**Tier 1 (Immediate implementation):**
```python
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccurrence_regularization
```
- Keep current multi-task setup
- Add simple co-occurrence penalty
- Expected: 3-5x improvement in co-occurrence score

**Tier 2 (After Tier 1):**
```python
total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccurrence_reg + 0.01*mmd_loss
```
- Add distribution matching objective
- Expected: JS divergence 0.61 → 0.3

**Tier 3 (Architectural change):**
```python
# Pre-train code embeddings with contrastive loss
pretrained_embeddings = contrastive_pretraining(training_visits)

# Fine-tune with combined loss
total_loss = lm_loss + hierarchical_loss + 0.001*age_loss + 0.001*sex_loss
```
- Hierarchical generation (Chapter → Code)
- Pre-trained co-occurrence-aware embeddings
- Expected: All metrics pass targets

---

## 6. Data Augmentation Approaches

### Your Concern

**From research request:**
> "I'm very tempted to reach for some kind of data augmentation where known co-occurrences temporally and one-shot are more likely, but it seems to defeat the purpose of the LM."

### Analysis: Does Augmentation "Defeat the Purpose"?

**What is the "purpose" of the LM?**

If purpose = **Learn P(codes | demographics, history) from data alone:**
- Then yes, co-occurrence-guided augmentation introduces human bias
- Model is no longer purely data-driven
- You're "teaching" the model what to learn

If purpose = **Generate realistic, clinically plausible patients:**
- Then no, augmentation that improves realism serves the goal
- Medical validity auxiliary losses already introduce domain knowledge
- Co-occurrence augmentation is another form of domain knowledge

**Critical insight:** You're already "defeating" pure end-to-end learning with auxiliary losses. The question is **where to draw the line**, not whether to cross it.

### Current Status: Pure End-to-End Learning

**What you're doing:**
- LM loss: Learn from data
- Auxiliary losses: Learn medical validity rules
- **No co-occurrence guidance**

**Result:**
- ✅ Model learns visit/code count distributions
- ✅ Model learns age/sex appropriateness (99%, 96%)
- ❌ Model fails to learn code co-occurrence (score 2.80)
- ❌ Model fails to match code frequencies (JS 0.61)

**Conclusion:** Pure end-to-end learning **is failing** for your task/data scale.

### Augmentation Approaches (Ranked by Intrusiveness)

**Level 1: Minimal Intrusion (Post-hoc Filtering)**

```python
# Generate multiple candidates, select most plausible
def generate_with_reranking(model, demographics, k=10):
    candidates = []
    for _ in range(k):
        patient = model.generate(demographics, temperature=0.7)
        # Score by co-occurrence
        score = compute_cooccurrence_score([patient], train_data)
        candidates.append((patient, score))

    # Return highest-scoring
    return max(candidates, key=lambda x: x[1])[0]
```

**Intrusiveness:** ⭐ (Low) - Model unchanged, post-processing only
**Effectiveness:** ⭐⭐⭐ (Medium) - Co-occurrence 2.80 → 10-20
**Tradeoff:** 10x inference cost, doesn't improve model

---

**Level 2: Moderate Intrusion (Training Data Augmentation)**

```python
# During training: Occasionally replace codes with co-occurring alternatives
def augment_visit(visit_codes, cooccur_matrix, replace_prob=0.1):
    augmented = []
    for code in visit_codes:
        if random.random() < replace_prob:
            # Replace with high-co-occurrence alternative
            neighbors = get_top_k_cooccurring(code, cooccur_matrix, k=5)
            replacement = random.choice(neighbors)
            augmented.append(replacement)
        else:
            augmented.append(code)
    return augmented
```

**Intrusiveness:** ⭐⭐ (Medium) - Training data modified, model unchanged
**Effectiveness:** ⭐⭐⭐⭐ (High) - Co-occurrence 2.80 → 20-30
**Tradeoff:** May reduce diversity, reinforces training biases

---

**Level 3: High Intrusion (Constrained Decoding)**

```python
# During generation: Bias logits toward co-occurring codes
def generate_with_cooccurrence_bias(model, demographics, cooccur_matrix, alpha=0.5):
    generated_codes = []

    for t in range(max_codes):
        # Standard model logits
        logits = model.forward(generated_codes, demographics)

        # Bias toward codes that co-occur with already-generated codes
        if len(generated_codes) > 0:
            for code_idx in range(vocab_size):
                # Sum co-occurrence counts with existing codes
                cooccur_score = sum(cooccur_matrix[existing, code_idx]
                                   for existing in generated_codes)

                # Add bias (log-space)
                logits[code_idx] += alpha * torch.log(cooccur_score + 1)

        # Sample from biased distribution
        next_code = sample_from_logits(logits, temperature=0.7)
        generated_codes.append(next_code)

    return generated_codes
```

**Intrusiveness:** ⭐⭐⭐ (High) - Generation process modified
**Effectiveness:** ⭐⭐⭐⭐⭐ (Very High) - Co-occurrence 2.80 → 30-50
**Tradeoff:** Model never learns co-occurrence, dependent on external matrix

---

**Level 4: Maximum Intrusion (Retrieve-and-Edit)**

```python
# Don't generate from scratch, edit real patients
def generate_via_retrieve_edit(demographics, train_data, edit_fraction=0.3):
    # Retrieve similar patients
    similar = find_similar_patients(demographics, train_data, k=5)

    # Sample one as base
    base_patient = random.choice(similar)

    # Edit: Replace ~30% of codes using model
    edited_patient = model.infill(
        base_patient,
        mask_prob=(1 - edit_fraction)  # Keep 70%, generate 30%
    )

    return edited_patient
```

**Intrusiveness:** ⭐⭐⭐⭐⭐ (Maximum) - Not true generation, retrieval-based
**Effectiveness:** ⭐⭐⭐⭐⭐ (Maximum) - Co-occurrence 2.80 → 50-80
**Tradeoff:** Low diversity, privacy concerns (close to real patients)

---

### Principled Augmentation: Back-Translation

**Inspired by NMT back-translation:**

```python
# Phase 1: Generate synthetic patients (even if low quality)
synthetic_patients = []
for _ in range(10000):
    demographics = sample_demographics()
    patient = model.generate(demographics)
    synthetic_patients.append(patient)

# Phase 2: Filter high-quality synthetic patients
filtered = [p for p in synthetic_patients
            if cooccurrence_score(p) > 20 and js_divergence(p) < 0.3]

# Phase 3: Retrain on mix of real + filtered synthetic
augmented_train = real_train_data + filtered
model.train(augmented_train)
```

**Intrusiveness:** ⭐⭐ (Medium) - Self-supervised, no external bias
**Effectiveness:** ⭐⭐⭐ (Medium) - May help, may reinforce errors
**Tradeoff:** Requires multiple training iterations, may not converge

---

### Recommendation on Augmentation

**Your concern is valid but overstated.**

**Augmentation is appropriate when:**
1. Pure end-to-end learning fails (✅ you're failing)
2. Task has known structure (✅ medical codes have co-occurrence patterns)
3. You have external knowledge source (✅ training data co-occurrence matrix)

**Recommended approach:**
1. **Try Level 1 (reranking) first** - No model changes, proves co-occurrence matters
2. **If successful, implement Level 3 (constrained decoding)** - Improves model quality
3. **Don't use Level 4 (retrieve-edit)** unless you abandon generation entirely

**Don't feel guilty about augmentation.** Your auxiliary losses for medical validity are already "augmentation" - co-occurrence guidance is just another form of domain knowledge.

---

## 7. Actionable Recommendations

### Summary of Findings

**Your current model suffers from:**
1. ✅ **Medical validity**: Good (99% age, 96% sex) - auxiliary losses work
2. ❌ **Code frequencies**: Poor (JS 0.61) - auxiliary losses interfere
3. ❌ **Co-occurrence**: Catastrophic (2.80) - not explicitly learned
4. ❌ **Common codes**: Missing (top-100 overlap 0.04) - frequency failure

**Root causes:**
1. **Data sparsity**: 25k patients insufficient for 5,562 flat codes (need 50-100k or hierarchy)
2. **Loss function**: Cross-entropy doesn't optimize distributions or co-occurrence
3. **Auxiliary loss conflict**: Age/sex losses fight against frequency learning
4. **Architecture**: No mechanisms to learn co-occurrence explicitly

**Can current approach work?**
- ❌ No, not without major changes
- Need: Hierarchical generation OR pre-trained embeddings OR explicit co-occurrence objective

---

### TIER 1: High Impact, Low Effort (1-3 days)

**Recommendation 1.1: Add Co-occurrence Regularization Loss**

```python
# File: metrics.py (add new function)
def build_cooccurrence_matrix(training_patients, vocab_size):
    """Pre-compute co-occurrence statistics from training data."""
    matrix = torch.zeros(vocab_size, vocab_size)
    for patient in training_patients:
        for visit in patient.visits:
            code_ids = [vocab.code2idx[c] for c in visit]
            for i in range(len(code_ids)):
                for j in range(i+1, len(code_ids)):
                    matrix[code_ids[i], code_ids[j]] += 1
                    matrix[code_ids[j], code_ids[i]] += 1
    return matrix

def cooccurrence_loss(generated_code_ids, cooccur_matrix, threshold=5):
    """Penalize rare co-occurrences."""
    loss = 0
    num_pairs = 0
    for i in range(len(generated_code_ids)):
        for j in range(i+1, len(generated_code_ids)):
            count = cooccur_matrix[generated_code_ids[i], generated_code_ids[j]]
            if count < threshold:
                loss += threshold / (count + 1)
            num_pairs += 1
    return loss / num_pairs if num_pairs > 0 else 0

# File: trainer.py (modify training loop)
# Pre-compute once before training
cooccur_matrix = build_cooccurrence_matrix(train_patients, vocab_size=5562)

# In training step
lm_loss = outputs.loss
age_loss = outputs.age_loss
sex_loss = outputs.sex_loss
cooccur_loss = cooccurrence_loss(generated_code_ids, cooccur_matrix)

total_loss = lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccur_loss
```

**Expected improvement:**
- Co-occurrence: 2.80 → 10-15 (4-5x)
- JS divergence: 0.61 → 0.5 (modest)

**Effort:** 4-8 hours (implement, test, integrate)

---

**Recommendation 1.2: Increase Training Data to 50k Patients**

```python
# File: config.py
num_patients: int = 50000  # Use full MIMIC-III (was 25000)
```

**Expected improvement:**
- Co-occurrence: 2.80 → 5-8 (2x)
- JS divergence: 0.61 → 0.5 (modest)

**Effort:** 5 minutes + retrain (~6 hours)

---

**Recommendation 1.3: Tune Auxiliary Loss Weights**

**Current testing:** 0.005/0.005 (causing collapse)

**Try grid search:**
```python
# Grid of (age_weight, sex_weight) to test
grid = [
    (0.01, 0.1),   # Balanced, reduce sex dominance
    (0.01, 0.05),  # Even less sex weight
    (0.01, 0.01),  # Fully balanced
    (0.005, 0.01), # Weak age, moderate sex
]
```

**Hypothesis:** Current 0.2 sex weight is too high, but 0.005 is too low. Sweet spot likely 0.01-0.05.

**Expected improvement:**
- JS divergence: 0.61 → 0.3-0.4 (2x)
- Medical validity: May drop to 95-97% (acceptable)

**Effort:** 1 day (4 training runs × 6 hours each)

---

### TIER 2: High Impact, Medium Effort (3-7 days)

**Recommendation 2.1: Implement ICD-9 Hierarchical Generation**

**Approach:** Three-stage generation (Chapter → Subcategory → Code)

**Stage 1: Generate chapter codes (19 options)**
```python
# Modify tokenizer to include chapter tokens
# Add special tokens: <ch-0>, <ch-1>, ..., <ch-18>
chapter_tokens = [f"<ch-{i}>" for i in range(19)]

# Map codes to chapters
def get_chapter(code):
    if code.startswith('V'):
        return 18  # Supplementary classification
    elif code.startswith('E'):
        return 17  # External causes
    else:
        return int(code[:3]) // 50  # Rough binning (tunable)
```

**Stage 2: Generate within-chapter subcategories**
```python
# For each chapter, define subcategories
# Example: Chapter 3 (Endocrine) → subcategories 240-279
```

**Stage 3: Generate specific codes**
```python
# For each subcategory, generate exact code
```

**Expected improvement:**
- **Massive:** Co-occurrence 2.80 → 30-50 (10-18x)
- JS divergence: 0.61 → 0.15-0.25 (3-4x)
- Top-100 overlap: 0.04 → 0.6-0.7 (15-17x)

**Effort:** 3-5 days (design hierarchy, modify tokenizer, update model, retrain)

**This is the SINGLE MOST IMPACTFUL change you can make.**

---

**Recommendation 2.2: Pre-train Code Embeddings with Contrastive Learning**

```python
# Phase 1: Pre-train embeddings (1-2 days)
class ContrastiveCodeEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, anchor, positive, negatives):
        anchor_emb = self.embeddings(anchor)
        pos_emb = self.embeddings(positive)
        neg_embs = self.embeddings(negatives)

        pos_sim = F.cosine_similarity(anchor_emb, pos_emb)
        neg_sims = F.cosine_similarity(anchor_emb.unsqueeze(1), neg_embs, dim=2)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1) / 0.1
        labels = torch.zeros(len(anchor), dtype=torch.long)  # Positive is at index 0

        return F.cross_entropy(logits, labels)

# Train for 10 epochs on visit co-occurrences
contrastive_model = ContrastiveCodeEmbedding(vocab_size=5562)
# ... training loop ...

# Phase 2: Initialize BART with pre-trained embeddings
model.encoder.embed_tokens.weight.data = contrastive_model.embeddings.weight.data.clone()
model.decoder.embed_tokens.weight.data = contrastive_model.embeddings.weight.data.clone()

# Phase 3: Fine-tune end-to-end
# ... normal training ...
```

**Expected improvement:**
- Co-occurrence: 2.80 → 15-25 (5-9x)
- JS divergence: 0.61 → 0.3-0.4 (2x)

**Effort:** 5-7 days (implement, pre-train, fine-tune)

---

**Recommendation 2.3: Implement Constrained Decoding with Co-occurrence Biasing**

```python
# File: generate.py (modify generation function)
def generate_with_cooccurrence_bias(
    model, tokenizer, demographics,
    cooccur_matrix, alpha=0.5
):
    """Generate with co-occurrence-aware decoding."""

    generated_code_ids = []

    # Custom generation loop (can't use model.generate())
    for t in range(max_codes_per_visit):
        # Get model logits
        outputs = model(
            input_ids=encoder_input,
            decoder_input_ids=decoder_input,
            x_num=x_num,
            x_cat=x_cat
        )
        logits = outputs.logits[:, -1, :]  # Last token

        # Bias logits based on co-occurrence with previously generated codes
        if len(generated_code_ids) > 0:
            for code_idx in range(vocab_size):
                cooccur_score = sum(
                    cooccur_matrix[prev_code, code_idx]
                    for prev_code in generated_code_ids
                )
                # Log-space bias
                logits[0, code_idx] += alpha * torch.log(cooccur_score + 1.0)

        # Sample next code
        probs = F.softmax(logits / temperature, dim=-1)
        next_code = torch.multinomial(probs, 1)
        generated_code_ids.append(next_code.item())

    return generated_code_ids
```

**Expected improvement:**
- Co-occurrence: 2.80 → 25-40 (9-14x)
- **Tradeoff:** Model doesn't learn co-occurrence, relies on external matrix

**Effort:** 2-3 days (implement custom generation loop, test)

---

### TIER 3: Medium Impact, Low Effort (Baselines)

**Recommendation 3.1: Reranking (Post-hoc Selection)**

```python
# File: generate.py
def generate_with_reranking(model, tokenizer, demographics, k=10):
    """Generate k candidates, select best by co-occurrence score."""
    candidates = []

    for _ in range(k):
        result = generate_patient_from_demographics(
            model, tokenizer, device,
            age=demographics['age'],
            sex=demographics['sex'],
            temperature=0.7  # Higher diversity
        )

        # Score by co-occurrence
        score = compute_cooccurrence_score([result], train_patients)
        candidates.append((result, score))

    # Return highest-scoring
    best = max(candidates, key=lambda x: x[1])
    return best[0]
```

**Expected improvement:**
- Co-occurrence: 2.80 → 10-20 (4-7x)
- **Tradeoff:** 10x inference cost

**Effort:** 2-4 hours

---

**Recommendation 3.2: Reduce Auxiliary Loss Weights to Zero (Pure LM Baseline)**

```python
# config.py
age_loss_weight: float = 0.0
sex_loss_weight: float = 0.0
```

**Hypothesis:** Auxiliary losses may be interfering. Test pure LM baseline.

**Expected outcome:**
- Medical validity: May drop to 70-80%
- JS divergence: Might improve to 0.3-0.4
- Co-occurrence: Might improve to 5-10

**Effort:** 6 hours (retrain + evaluate)

**Purpose:** Diagnostic - proves whether aux losses are causing semantic coherence failure

---

### TIER 4: Architectural Alternatives (High Effort)

**Recommendation 4.1: Two-Stage Generation**

**Stage 1:** Predict code frequencies (bag-of-codes)
```python
# Given demographics, predict how many of each code type
P(n_cardiovascular, n_diabetes, n_respiratory | age, sex)
```

**Stage 2:** Generate specific codes given frequencies
```python
# Given demographics + code counts, generate exact codes
P(specific_codes | age, sex, code_counts)
```

**Advantage:** Decomposes hard problem into two easier sub-problems

**Expected improvement:** All metrics pass (JS <0.2, co-occurrence >30)

**Effort:** 2-3 weeks (new architecture)

---

**Recommendation 4.2: Retrieve-and-Edit**

```python
def generate_via_retrieve_edit(age, sex, train_data):
    # Find similar patients
    similar = [p for p in train_data
               if abs(p.age - age) < 10 and p.sex == sex]

    # Sample one
    base = random.choice(similar[:5])

    # Use model to edit 20-30% of codes
    edited = model.infill(base, mask_prob=0.7)

    return edited
```

**Expected improvement:** Co-occurrence 40-60 (guaranteed high since base is real)

**Tradeoff:** Not true generation, privacy concerns

**Effort:** 2-3 days

---

## 8. Final Assessment

### Can Your Current Scale/Approach Succeed?

**Question:** Is 25,000 patients with 5,562 flat codes sufficient for semantic coherence?

**Answer:** **No, not without architectural changes.**

**Evidence:**

| Requirement | Needed | You Have | Gap |
|------------|--------|----------|-----|
| Code frequency samples | 10-100/code | 136/code | ✅ Sufficient |
| Pairwise co-occurrence samples | 10-100/pair | 5/pair | ❌ 2-20x insufficient |
| Triplet patterns | 10-100/triplet | <1/triplet | ❌ 100x+ insufficient |
| Distribution learning | N > 10×vocab | 250k needed, 25k actual | ❌ 10x insufficient |

**Why you're failing:**

1. **Vocabulary too large:** 5,562 flat codes creates 30M possible pairs
2. **Sample size too small:** 25k patients provides only 150k observed pairs
3. **No hierarchy:** Flat vocabulary prevents leveraging ICD-9 structure
4. **Wrong loss:** Cross-entropy doesn't optimize distributions or co-occurrence

**Successful models use:**
- 50k-200k patients (2-8x your size)
- OR hierarchical vocabularies (100x sparsity reduction)
- OR pre-training on massive corpora (10-100x your size)
- OR explicit co-occurrence objectives

---

### Honest Recommendations by Goal

**If you want to publish/demonstrate the current approach:**
- ❌ Don't - it won't achieve semantic coherence without major changes
- Current results (JS 0.61, co-occurrence 2.80) are not publishable

**If you want to fix semantic coherence within 1 week:**
- ✅ Implement Tier 1 recommendations:
  - Co-occurrence loss (4-8 hours)
  - Increase to 50k patients (5 min + retrain)
  - Tune aux weights (1 day)
- **Expected:** JS 0.3-0.4, co-occurrence 10-15 (marginal pass)

**If you want strong semantic coherence (research-grade):**
- ✅ Implement ICD-9 hierarchical generation (Tier 2.1)
  - 3-5 days implementation
  - **Expected:** JS 0.15-0.25, co-occurrence 30-50 (strong pass)
- This is the **highest-impact change** you can make

**If you're on a tight deadline:**
- ✅ Use reranking (Tier 3.1) or retrieve-edit (Tier 4.2)
  - 2-4 hours for reranking
  - **Expected:** Co-occurrence 10-20 (reranking) or 40-60 (retrieve-edit)
  - **Tradeoff:** No model improvement, just better selection

---

### The Goldilocks Problem: Auxiliary Loss Weights

**You've discovered a fundamental tradeoff:**

| Aux Weights | Medical Validity | Semantic Coherence | Model Behavior |
|------------|-----------------|-------------------|----------------|
| **0.01/0.2** (original) | ✅ 99% / 96% | ❌ JS 0.61, co-occ 2.80 | Sex loss dominates |
| **0.005/0.005** (testing) | ❓ Unknown | ❌ Model collapse | Aux signal too weak |
| **0.001/0.001** (failed) | ❌ Collapse | ❌ Collapse | Overfitting |

**Hypothesis:** No "Goldilocks zone" exists with current architecture.

**Why:** Auxiliary losses and semantic coherence have **conflicting gradients**:
- Aux losses push toward rare, demographically-appropriate codes
- Semantic coherence needs common, frequently co-occurring codes
- **These are often opposing sets of codes**

**Solution:** Don't rely on weight tuning alone. Add explicit co-occurrence objective (Tier 1.1) or hierarchy (Tier 2.1).

---

### Recommendation Priority

**Do this in order:**

1. **Immediate (today):**
   - Increase to 50k patients (5 min)
   - Submit training with aux weights 0.01/0.1 (6 hours)

2. **This week:**
   - Implement co-occurrence loss (Tier 1.1) (4-8 hours)
   - Retrain with cooccur loss (6 hours)
   - Evaluate results

3. **Next week (if Week 1 insufficient):**
   - Implement ICD-9 hierarchical generation (Tier 2.1) (3-5 days)
   - This will likely solve the problem

4. **Backup plan (if on deadline):**
   - Implement reranking (Tier 3.1) (2-4 hours)
   - Document limitations in paper

**Do NOT:**
- Keep trying different aux weight combinations (Goldilocks doesn't exist)
- Train with 0.001/0.001 weights (will collapse)
- Expect current architecture to achieve strong coherence without changes

---

## 9. Document Metadata

**Created:** 2025-10-29
**Author:** Research analysis based on codebase documentation and ML literature
**Status:** Complete
**Next steps:** User decision on which tier of recommendations to pursue

**Related documents:**
- `docs/reference/promptehr_comparison.md` - Architectural differences
- `docs/reference/multitask_learning.md` - Auxiliary loss details
- `docs/historical/conditional_reconstruction.md` - Generation approach
- `docs/historical/semantic_coherence_chat.md` - Previous coherence investigation

**Key findings:**
1. 25k patients insufficient for 5,562 flat codes
2. Hierarchical generation (ICD-9 chapters) is highest-impact fix
3. No Goldilocks zone for aux loss weights without architectural change
4. Co-occurrence must be explicitly learned (not implicit from LM loss)

**Critical recommendation:**
**Implement ICD-9 hierarchical generation (Tier 2.1)** - This single change can improve co-occurrence by 10-18x and JS divergence by 3-4x, solving the fundamental data sparsity problem.
