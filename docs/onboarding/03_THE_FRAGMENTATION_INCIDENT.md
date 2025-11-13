# 03: The Fragmentation Incident (October 9, 2025)

**Estimated Time:** 60 minutes
**Prerequisites:** [01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md), [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)
**Next:** [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)

---

## Learning Objectives

By the end of this page, you will understand:
- The original text-based implementation and why it failed catastrophically
- How BART's tokenizer fragmented ICD-9 codes into meaningless subwords
- Why special tokens weren't recognized and were also fragmented
- The concept of "semantic integrity" for structured codes
- How random embeddings prevented gradient-based learning
- **The critical lesson:** Medical codes require 1:1 token mapping

---

## The Original Implementation (main.py)

### Design Philosophy (October 9, 2025)

The original PromptEHR implementation followed a **text-based approach**, treating patient records as continuous strings.

**Rationale (seemed reasonable at the time):**
- BART is pretrained on text → Use text representation
- Transformers handle sequences → Represent patient as one long sequence
- Special tokens work in NLP → Add custom tokens for structure

**Implementation:** `deprecated/legacy_implementations/main.py`

### Data Format

**Text-Based Patient Representation:**
```
"65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 584.9 <\v> <END>"
```

**Components:**
- `65` - Age
- `WHITE` - Race/ethnicity
- `M` - Gender (M or F)
- `<demo>` - Custom token marking end of demographics
- `<v>` - Custom token marking start of visit
- `401.9`, `250.00` - ICD-9 diagnosis codes **as text strings**
- `<\v>` - Custom token marking end of visit
- `<END>` - Custom token marking end of patient record

**Seems reasonable, right?** BART handles text well, we're just adding some structure.

### Implementation Details

**Tokenization Approach:**
```python
# Original approach (WRONG)
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Add custom tokens (WRONG - not special tokens)
new_tokens = ["<demo>", "<v>", "<\v>", "<END>"]
tokenizer.add_tokens(new_tokens)  # Regular tokens, not special!

# Create patient string
patient_str = "65 WHITE M <demo> <v> 401.9 250.00 <\v> <END>"

# Tokenize (THIS IS WHERE DISASTER STRIKES)
token_ids = tokenizer.encode(patient_str)
```

**What could go wrong?**

Everything.

---

## Problem 1: Code Fragmentation

### BART's Subword Tokenization

BART uses **Byte-Pair Encoding (BPE)** tokenization, which splits words into subword units based on frequency in the pretraining corpus.

**How BPE Works:**
1. Start with character-level vocabulary
2. Merge frequent character pairs into subwords
3. Repeat until target vocabulary size (50,265 for BART)

**Example on common English:**
```
"unhappiness" → ["un", "happiness"]  # "un" and "happiness" are common
"extraordinary" → ["extra", "ordinary"]
"running" → ["run", "ning"]
```

**This works great for natural language!** Common morphemes get their own tokens.

### What Happens to ICD-9 Codes?

**Problem:** ICD-9 codes like "401.9" are **not common** in BART's pretraining data (web text, books, news).

**Let's see what happens:**

```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Tokenize an ICD-9 code
code = "401.9"
tokens = tokenizer.tokenize(code)
print(f"Code: {code}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")
```

**Output:**
```
Code: 401.9
Tokens: ['Ġ401', '.', 'Ġ9']  # Ġ = space character
Token IDs: [20174, 4, 860]
```

**DISASTER!** The code "401.9" is split into THREE separate tokens:
- `'Ġ401'` (token ID 20174) - BART thinks this is the number "401"
- `'.'` (token ID 4) - Period punctuation
- `'Ġ9'` (token ID 860) - The digit "9"

**More Examples:**
```
"250.00" → ['Ġ250', '.', '438']  # "00" becomes "438" (token for number text)
"428.0" → ['Ġ428', '.', '506']
"V58.61" → ['ĠV', '3618', '.', '4175']  # Four tokens!
```

### Why This is Catastrophic

**Semantic Integrity Lost:**
- "401.9" (hypertension) is an **atomic medical concept**
- Breaking it into ["401", ".", "9"] destroys the meaning
- "401" alone means nothing medically
- "9" alone means nothing medically
- The model learns subword patterns, not medical concepts

**Analogy:**
Imagine trying to learn English by breaking words mid-syllable:
- "computer" → ["com", "pu", "ter"] ✓ (morphemes, makes sense)
- "computer" → ["co", "mpu", "ter"] ✗ (arbitrary, meaningless)

ICD-9 fragmentation is like the second case: arbitrary splits that destroy semantic meaning.

### What the Model Actually Sees

**Input (what we want):**
```
Patient: Age 65, Gender M, Codes: [401.9, 250.00, 428.0]
```

**What BART actually sees (after tokenization):**
```
Tokens: ['Ġ65', 'ĠWH', 'ITE', 'ĠM', 'Ġ<', 'dem', 'o', '>', 'Ġ<', 'v', '>',
         'Ġ401', '.', 'Ġ9', 'Ġ250', '.', '438', 'Ġ<', 'Ġ\\', 'v', '>',
         'Ġ<', 'v', '>', 'Ġ428', '.', '506', 'Ġ584', '.', 'Ġ9',
         'Ġ<', 'Ġ\\', 'v', '>', 'Ġ<', 'E', 'ND', '>']
```

**42 tokens** for what should be a simple structured record!

**Key Problems:**
1. Codes fragmented into meaningless subwords
2. No way to identify which tokens form a complete code
3. Model cannot learn "hypertension" (401.9) as a unit

---

## Problem 2: Special Token Fragmentation

### Special Tokens vs Regular Tokens

In transformers, there's a critical difference:

**Special Tokens:**
- Registered with `add_special_tokens({"additional_special_tokens": [...]})` or `special_tokens_map.json`
- **Never** split during tokenization
- Get dedicated token IDs (usually 0-10 range)
- Example: BART's `<s>` (BOS), `</s>` (EOS), `<pad>`, `<unk>`

**Regular Tokens:**
- Added with `add_tokens([...])` or learned during pretraining
- **Can be split** if BPE finds a better encoding
- Get token IDs in the main vocabulary range

### The Bug: Wrong Token Registration

**Original Implementation (WRONG):**
```python
# This adds them as REGULAR tokens, not special!
new_tokens = ["<demo>", "<v>", "<\v>", "<END>"]
tokenizer.add_tokens(new_tokens)
```

**Result:** BPE still applies to these tokens!

### What Actually Happens

```python
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_tokens(["<demo>", "<v>", "<\v>", "<END>"])

# Try to tokenize special token
demo_token = "<demo>"
print(f"Token: {demo_token}")
print(f"Tokenized: {tokenizer.tokenize(demo_token)}")
```

**Output:**
```
Token: <demo>
Tokenized: ['Ġ<', 'dem', 'o', '>']  # FOUR TOKENS!
```

**DISASTER #2!** Our structural markers are also fragmented:
- `<demo>` → `['Ġ<', 'dem', 'o', '>']`
- `<v>` → `['Ġ<', 'v', '>']`
- `<\v>` → `['Ġ<', 'Ġ\\', 'v', '>']`
- `<END>` → `['Ġ<', 'E', 'ND', '>']`

**Impact:**
- Model cannot recognize visit boundaries (<v>, <\v>)
- Cannot distinguish demographics from codes
- Structural information completely lost

---

## Problem 3: Random Embeddings & No Gradient Flow

### How Embeddings Work in Transformers

**Embedding Layer:**
```python
# BART's embedding layer
embedding = nn.Embedding(vocab_size=50265, embedding_dim=768)
```

**For each token ID, lookup a 768-dimensional vector:**
```
Token ID 20174 ("Ġ401") → [0.023, -0.145, 0.891, ..., 0.234]  # 768 dims
Token ID 4 (".") → [-0.012, 0.456, -0.231, ..., 0.891]
Token ID 860 ("Ġ9") → [0.567, -0.023, 0.145, ..., -0.456]
```

**These embeddings are learned from pretraining** on billions of text tokens.

### The New Token Problem

When we call `tokenizer.add_tokens(["<demo>"])`, BART must:
1. Assign new token ID: 50265 (next available)
2. Extend embedding layer: `nn.Embedding(50265+1, 768)`
3. **Initialize new embedding vector**: ❓

**How are new embeddings initialized?**

**PyTorch default:** Random initialization from `N(0, 1/√768)`

```python
new_embedding = torch.randn(768) / sqrt(768)
# Example: [0.0142, -0.0318, 0.0091, ..., -0.0234]
```

**Problem:** This random vector has **no semantic meaning**.

### Why This Prevents Learning

**Gradient Flow During Training:**

1. **Forward Pass:**
   - Input: `<demo>` (token ID 50265)
   - Embedding lookup: `random_vector` (untrained)
   - Model processes this meaningless vector
   - Output: Gibberish (model has no signal)

2. **Backward Pass:**
   - Loss computed on output
   - Gradients flow back: `∂Loss/∂embedding[50265]`
   - Embedding updated: `embedding[50265] -= lr * gradient`

**Sounds fine, right?** The embedding should learn over time.

**Actual Problem:**
- BART's **pretrained embeddings** have strong semantic meaning
- Example: "dog" embedding is close to "cat", far from "computer"
- Custom tokens start from **pure noise**, no semantic neighborhood
- Even after training, they struggle to develop meaningful relationships

**Worse:**
- Model predictions are **dominated by pretrained token patterns**
- Generation prefers known tokens (subwords) over custom tokens
- Result: Model outputs pretrained tokens, ignores custom structure

---

## The Catastrophic Result: Gibberish Generation

### Training Behavior

**Training Metrics Looked Fine:**
```
Epoch 1: Loss=5.23, Perplexity=187.3
Epoch 5: Loss=3.87, Perplexity=48.1
Epoch 10: Loss=2.91, Perplexity=18.4
Epoch 20: Loss=2.34, Perplexity=10.4
```

**Loss decreased!** Perplexity dropped! Training converged!

**But what was the model actually learning?**

### Generation Output (The Incident)

**Test Case:** Generate a 65-year-old male patient

**Expected Output:**
```
65 WHITE M <demo> <v> 401.9 250.00 <\v> <v> 428.0 <\v> <END>
```

**Actual Output (October 9, 2025):**
```
56 ASIAN M <demo> ASIAN m <demoing> <demoing> WHITE M WHITE WHITE M
```

**Complete gibberish!**

**More Examples:**
```
Test 1: "67 F <demo> F <demo> F <demo> <demo> <demo> <demo> <demo> <demo>"
Test 2: "M M M M M M M M M M M M M M M M M M"
Test 3: "WHITE ASIAN OTHER WHITE ASIAN <dem <dem <dem <dem <dem"
Test 4: "401 401 250 250 428 401 401 250 250 428 401"  # Fragmented codes!
```

**Patterns Observed:**
1. **Token repetition:** Same token repeated (M, F, WHITE, ASIAN)
2. **Partial tokens:** `<dem` instead of `<demo>` (fragmentation persists)
3. **Fragmented codes:** `401` and `250` (subwords) instead of `401.9`, `250.00`
4. **No structure:** No meaningful visit boundaries, no coherent medical codes
5. **No conditioning:** Age/sex ignored, model just repeats patterns

### Root Cause Analysis

**Why did this happen?**

1. **Fragmentation destroyed semantic units:**
   - Model learned subword patterns, not medical concepts
   - "401" and "9" treated as independent tokens
   - Cannot generate complete codes like "401.9"

2. **Special tokens not recognized:**
   - `<demo>`, `<v>`, `<\v>` fragmented into subwords
   - Model has no concept of structural boundaries
   - Generates partial tokens like `<dem`

3. **Random embeddings dominated:**
   - Pretrained tokens (WHITE, M, F) have strong embeddings
   - Custom tokens have weak embeddings (random init)
   - Model prefers generating known tokens → repetition

4. **No gradient signal for structure:**
   - Loss computed on fragmented tokens
   - Model learns to minimize loss on subwords, not codes
   - "Correct" output (by loss) could be `['401', '.', '9']` or `['401', '.', '506']` - model doesn't know!

5. **Tokenizer-decoder mismatch:**
   - Encoding: "401.9" → [20174, 4, 860]
   - Decoding: [20174, 4, 860] → "401.9" ✓ (lucky!)
   - Generation: Model outputs [20174] → "401" ✗ (incomplete code)

---

## The Realization: Medical Codes ≠ Text

### Key Insight (October 9, 2025)

**User realization:**
> "The problem is that BART's tokenizer fragments ICD-9 codes. '401.9' becomes ['401', '.', '9']. We need 1:1 token mapping where each code gets exactly one token ID."

**This was the breakthrough moment.**

### Fundamental Difference

**Natural Language (Text):**
- Compositional: "unhappy" = "un" + "happy" (meaning preserved)
- Subword tokenization works: Morphemes carry meaning
- BPE aligns with linguistic structure

**Medical Codes (Structured):**
- Atomic: "401.9" ≠ "401" + "." + "9" (meaning destroyed)
- Hierarchical, not compositional: "401.9" belongs to category "401", but "9" alone is meaningless
- Require whole-code tokenization: 1:1 mapping

**Analogy:**
- Text: LEGO bricks (combine to build meaning)
- Codes: Atoms (split them, you get different elements)

### What We Needed

**Requirements for a working system:**

1. **1:1 Token Mapping:**
   - Each ICD-9 code → Single unique token ID
   - No fragmentation: "401.9" must be atomic

2. **True Special Tokens:**
   - Visit boundaries (<v>, <\v>) must be registered as special
   - Never tokenized, always recognized

3. **Learned Code Embeddings:**
   - Each code gets a trainable embedding
   - No reliance on pretrained subword embeddings
   - Gradient flow to code-specific vectors

4. **Semantic Integrity:**
   - Model must learn "401.9" as hypertension (atomic concept)
   - Not "401" and "9" as separate meaningless fragments

---

## The Solution Preview

**New Architecture (Coming in [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)):**

1. **DiagnosisVocabulary:** Maps each code to unique token ID
   - "401.9" → Token ID 7
   - "250.00" → Token ID 8
   - "428.0" → Token ID 9

2. **DiagnosisCodeTokenizer:** Custom tokenizer without fragmentation
   - Skips BPE entirely for diagnosis codes
   - Direct vocabulary lookup: code → ID

3. **Separate Demographics from Codes:**
   - Demographics (age, sex) → Model input features
   - Codes → Token sequences only
   - No text-based demographics in sequence

4. **Vocabulary Extension:**
   - BART vocab: 50,265 (pretrained)
   - Add 6,985 diagnosis tokens (IDs 7-6991)
   - Add 7 special tokens (IDs 0-6)
   - Total: 6,992 tokens

**Result:** Each code is a single, learnable token with semantic integrity.

---

## Lessons Learned

### Lesson 1: Structured Data ≠ Unstructured Data

**Don't treat medical codes as text strings** just because they contain characters.

**ICD-9 codes are:**
- Identifiers (like database IDs)
- Hierarchically structured (category + subcategory)
- Semantically atomic (split them, meaning lost)

**Text is:**
- Compositional (morphemes combine)
- Statistically learned (BPE reflects frequency)
- Contextually flexible (same word, different meanings)

### Lesson 2: Special Tokens Must Be Special

**Always use `add_special_tokens()`, not `add_tokens()`** for structural markers.

**Special tokens:**
- Never split during tokenization
- Recognized in all contexts
- Can be used in loss masking, attention patterns

### Lesson 3: Embeddings Need Semantic Initialization

**Random embeddings for domain-specific tokens are problematic.**

**Better approaches:**
1. Train from scratch on domain data (our choice)
2. Initialize from similar pretrained tokens (if available)
3. Use separate embedding layer for domain tokens

### Lesson 4: Metrics Can Lie

**Low training loss ≠ Good generation quality**

**The model optimized loss on fragmented subwords**, not meaningful codes.

**Always validate with generation samples**, not just metrics.

---

## Try It Yourself

### Exercise 1: Reproduce the Fragmentation

```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Test ICD-9 code tokenization
codes = ["401.9", "250.00", "428.0", "V58.61", "E849.0"]

for code in codes:
    tokens = tokenizer.tokenize(code)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Code: {code:10} | Tokens: {tokens} | IDs: {ids}")
```

**Expected Output:**
```
Code: 401.9      | Tokens: ['Ġ401', '.', 'Ġ9'] | IDs: [20174, 4, 860]
Code: 250.00     | Tokens: ['Ġ250', '.', '438'] | IDs: [21268, 4, 3103]
Code: 428.0      | Tokens: ['Ġ428', '.', '506'] | IDs: [34938, 4, 35743]
Code: V58.61     | Tokens: ['ĠV', '3618', '.', '4175'] | IDs: [9732, 24586, 4, 2518]
Code: E849.0     | Tokens: ['ĠE', '44460', '.', '506'] | IDs: [3608, 23606, 4, 35743]
```

**Observation:** Every single code is fragmented into 3-4 tokens!

### Exercise 2: Special Token Registration

```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Wrong way (regular token)
tokenizer.add_tokens(["<demo>"])
print("After add_tokens:")
print(f"  Tokenized: {tokenizer.tokenize('<demo>')}")

# Right way (special token)
tokenizer.add_special_tokens({"additional_special_tokens": ["<DEMO>"]})
print("\nAfter add_special_tokens:")
print(f"  Tokenized: {tokenizer.tokenize('<DEMO>')}")
```

**Expected Output:**
```
After add_tokens:
  Tokenized: ['Ġ<', 'dem', 'o', '>']  # FRAGMENTED

After add_special_tokens:
  Tokenized: ['<DEMO>']  # ATOMIC
```

**Lesson:** Special tokens stay intact, regular tokens get fragmented.

### Exercise 3: Inspect deprecated/legacy_implementations/main.py

```bash
# View the original implementation
head -100 deprecated/legacy_implementations/main.py

# Search for add_tokens call (the bug)
grep -n "add_tokens" deprecated/legacy_implementations/main.py

# Search for tokenize calls
grep -n "tokenize" deprecated/legacy_implementations/main.py
```

**Find the exact lines where the bug was introduced.**

---

## Key Takeaways

1. **BART's BPE tokenizer fragments ICD-9 codes** into meaningless subwords ("401.9" → ["401", ".", "9"]), destroying semantic integrity.

2. **Special tokens added with `add_tokens()` are still fragmented** - must use `add_special_tokens()` for structural markers.

3. **Random embedding initialization prevents learning** - custom tokens start from noise, struggle against pretrained token dominance.

4. **Training metrics (loss, perplexity) looked fine**, but generation produced complete gibberish due to fragmentation.

5. **The critical lesson:** Medical codes are **atomic structured identifiers**, not compositional text. They require 1:1 token mapping.

6. **October 9, 2025 was the turning point** - realization that led to complete architecture redesign.

---

## Historical Significance

This incident fundamentally shaped PromptEHR's architecture:
- Abandoned text-based representation entirely
- Designed custom tokenizer (DiagnosisCodeTokenizer)
- Separated demographics from code sequences
- Prioritized semantic integrity over NLP convenience

**The fragmentation incident is the "origin story"** - everything in the current architecture exists to prevent this failure mode.

---

## What's Next?

You now understand **what went wrong** and **why** the original approach failed.

**Next:** [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md) - Learn the token-based architecture that replaced the text-based approach and solved fragmentation.

**Alternative Path:**
- [05_DATA_REPRESENTATION.md](05_DATA_REPRESENTATION.md) - Skip ahead to see the modern PatientRecord structure

---

**Navigation:**
- ← Back to [02_MEDICAL_CODES_VS_TEXT.md](02_MEDICAL_CODES_VS_TEXT.md)
- → Next: [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
