# 02: Medical Codes vs Natural Language Text

**Estimated Time:** 45 minutes
**Prerequisites:** [01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md)
**Next:** [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md)

---

## Learning Objectives

By the end of this page, you will understand:
- Why medical codes are fundamentally different from natural language
- The concept of "semantic integrity" for structured identifiers
- Why subword tokenization (BPE) fails for medical codes
- The difference between compositional and atomic meaning
- Why treating codes as text leads to catastrophic failures

---

## The Naive Assumption

When you first approach EHR generation, a natural thought is:

> "Medical codes are just text strings. BART is great at text. Let's feed codes to BART!"

**This seems reasonable:**
- ICD-9 codes contain alphanumeric characters: "401.9", "V58.61"
- BART has a text tokenizer (BPE)
- BART generates text successfully
- Therefore, BART should generate codes successfully

**This assumption is catastrophically wrong.** Here's why.

---

## Natural Language: Compositional Meaning

### How Natural Language Works

Natural language is **compositional** - meaning emerges from combining smaller units (morphemes, words).

**Examples:**

```
"unhappy" = "un" (not) + "happy" (joyful)
         → "not joyful"

"running" = "run" (action) + "ing" (present participle)
         → "currently performing the action of running"

"extraordinary" = "extra" (beyond) + "ordinary" (normal)
                → "beyond normal"
```

**Key Property:** You can break words into parts and **meaning is preserved** (or predictably transformed).

### Why BPE Works for Natural Language

**Byte-Pair Encoding (BPE)** tokenization exploits compositionality:

1. Start with character-level vocabulary
2. Merge frequent character pairs
3. Build subword vocabulary based on corpus statistics

**Example:**
```
Corpus: "running runner ran run"

After BPE:
"running" → ["runn", "ing"]  # "runn" is frequent subword
"runner"  → ["runn", "er"]
"ran"     → ["ran"]
"run"     → ["run"]
```

**Why this works:**
- "runn" captures the root meaning (locomotion)
- "ing", "er" capture morphological meaning (tense, agent)
- Model learns: "runn" + "ing" = present participle verb

**Linguistic Alignment:** BPE boundaries align with morpheme boundaries (often).

---

## Medical Codes: Atomic Meaning

### How Medical Codes Work

Medical codes are **structured identifiers**, not compositional text.

**ICD-9 Code Format:**
```
XXX.XX

401.9    = Essential hypertension, unspecified
401.0    = Malignant hypertension
401.1    = Benign hypertension

250.00   = Diabetes mellitus type II, no complications
250.01   = Diabetes mellitus type I, no complications
```

### The Structure is Hierarchical, Not Compositional

**ICD-9 Hierarchy:**

```
400-459: Diseases of the Circulatory System (Chapter)
  ├─ 401-405: Hypertensive Disease (Section)
  │   ├─ 401: Essential Hypertension (Category)
  │   │   ├─ 401.0: Malignant
  │   │   ├─ 401.1: Benign
  │   │   └─ 401.9: Unspecified
  │   └─ 402: Hypertensive Heart Disease
  └─ 410-414: Ischemic Heart Disease
```

**Key Property:** The code "401.9" is an **atomic identifier** for a specific medical concept.

**Breaking it down destroys meaning:**
- "401" alone = Category (hypertensive disease) - **not a diagnosis**
- "9" alone = Subcategory marker - **meaningless without category**
- "." = Delimiter - **no semantic content**

**Analogy:**
- Natural language: LEGO bricks (combine to build meaning)
- Medical codes: Chemical elements (split the atom, you get different elements)

---

## Why BPE Fails for Medical Codes

### BPE Has No Medical Knowledge

**BPE tokenization statistics (from BART pretraining):**

- Corpus: Books, web text, news articles (billions of tokens)
- "401.9" appears: **Never** or **extremely rarely**
- "401" as a number: Occasionally ("401(k) plan", "HTTP 401 error")
- "9" as a digit: Very frequently

**Result:** BPE treats "401.9" as arbitrary character sequence, not medical code.

### Actual BPE Tokenization

**Code:** "401.9"

**BART's BPE tokenization:**
```python
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokens = tokenizer.tokenize("401.9")
# Output: ['Ġ401', '.', 'Ġ9']  (Ġ = space character)
```

**Three separate tokens:**
1. `'Ġ401'` (token ID 20174) - pretrained embedding for number "401"
2. `'.'` (token ID 4) - pretrained embedding for period punctuation
3. `'Ġ9'` (token ID 860) - pretrained embedding for digit "9"

**Problem:** Model has three independent embedding vectors with **no knowledge they form a medical code**.

---

## Semantic Integrity: Atomic vs Compositional

### Definition

**Semantic Integrity** = Whether meaning is preserved when a unit is split into parts.

**Natural Language (Compositional):**
- "unhappy" → ["un", "happy"] ✓ **Meaning preserved** (not + joyful)
- "running" → ["run", "ning"] ✓ **Meaning preserved** (action + present)

**Medical Codes (Atomic):**
- "401.9" → ["401", ".", "9"] ✗ **Meaning destroyed** (category + delimiter + marker ≠ hypertension diagnosis)
- "V58.61" → ["V", "58", ".", "61"] ✗ **Meaning destroyed** (four meaningless fragments)

### What Happens When Semantic Integrity is Violated

**Training:**
- Model learns: "401" often followed by "."
- Model learns: "." often followed by "9"
- Model learns: These patterns correlate with patient age, comorbidities

**But model does NOT learn:**
- "401.9" is a **single atomic concept** (essential hypertension)
- "401.9" has semantic relationship with "428.0" (heart failure co-occurs)

**Generation:**
- Model generates: "401" → "." → "506" (following learned patterns)
- Result: "401.506" - **INVALID CODE** (doesn't exist in ICD-9)
- Or generates: "401" → " " → "250" (two separate codes, not "401.250")

**The model cannot guarantee generated subwords form valid codes.**

---

## Concrete Example: What the Model "Learns"

### Training Scenario

**Patient 1:** Age 65, Codes: ["401.9", "250.00", "428.0"]
**Patient 2:** Age 70, Codes: ["401.9", "585.9"]

**After BPE:**
```
Patient 1: ['Ġ401', '.', 'Ġ9', 'Ġ250', '.', '00', 'Ġ428', '.', 'Ġ0']
Patient 2: ['Ġ401', '.', 'Ġ9', 'Ġ585', '.', 'Ġ9']
```

**What model learns (simplified):**
```
P("." | "401") = 1.0         # "401" always followed by "."
P("9" | "401.") = 0.8        # "." after "401" often followed by "9"
P("250" | "9") = 0.3         # "9" sometimes followed by "250"
P("." | "250") = 1.0         # "250" always followed by "."
```

**Generation for Age 65:**
```
Sample: "401" → "." → "9" → "250" → "." → "00"
Decoded: "401.9 250.00"  ✓ Valid (lucky!)

Sample: "401" → "." → "9" → "585" → "." → "506"
Decoded: "401.9 585.506"  ✗ Invalid! (585.506 doesn't exist)

Sample: "401" → "428" → "." → "0"
Decoded: "401428.0"  ✗ Invalid! (wrong grouping)
```

**Problem:** Model learns subword transition probabilities, not code-level semantics.

---

## Why This Matters for EHR Generation

### Real-World Impact

**Medical codes must be:**
1. **Valid:** Exist in ICD-9 vocabulary (6,985 specific codes)
2. **Medically plausible:** Age/sex appropriate
3. **Semantically coherent:** Co-occur realistically

**BPE tokenization violates requirement #1:**
- Generated subwords may not form valid codes
- No guarantee "401" + "." + "506" = valid ICD-9 code
- Decoding is ambiguous (where do codes start/end?)

### Example: Generation Failure

**Input:** Generate patient, age 67, male

**With BPE (fragmentation):**
```
Generated tokens: ['Ġ401', '.', '506', 'Ġ250', 'Ġ428', '.', 'Ġ0']

Decoding attempt 1: "401.506 250428.0"    ✗ Both invalid
Decoding attempt 2: "401.506 250 428.0"   ✗ 401.506 invalid, 250 needs decimal
Decoding attempt 3: "401.50 6250 428.0"   ✗ Nonsense
```

**With 1:1 token mapping (our solution):**
```
Generated tokens: [401.9, 250.00, 428.0]  (token IDs: [7, 8, 9])

Decoding: "401.9 250.00 428.0"  ✓ All valid codes
```

**Key Insight:** 1:1 mapping guarantees generated tokens are valid codes.

---

## The Fundamental Difference

### Natural Language Generation

**Task:** Generate coherent English sentence

**Input:** "The patient is"
**Model generates:** ["Ġa", "Ġ65", "-", "year", "-", "old", "Ġmale"]
**Decoded:** "The patient is a 65-year-old male" ✓

**Why it works:**
- Subwords compose into valid words ("year-old")
- Grammar emerges from subword patterns
- Rare words can be constructed from known subwords

### Medical Code Generation (with BPE)

**Task:** Generate valid ICD-9 codes

**Input:** Age 65, Male
**Model generates:** ['Ġ401', '.', 'Ġ9', 'Ġ250', '.', '00']
**Decoded:** "401.9 250.00" ✓ **Valid** (but only by chance!)

**Next generation:**
**Model generates:** ['Ġ401', '.', '506', 'Ġ250', '.', '123']
**Decoded:** "401.506 250.123" ✗ **Invalid codes!**

**Why it fails:**
- Subwords do NOT compose into valid codes
- Code validity requires exact vocabulary match
- Cannot construct new codes from fragments

---

## Comparison Table

| Property | Natural Language | Medical Codes |
|----------|-----------------|---------------|
| **Meaning Type** | Compositional | Atomic |
| **Unit Breakdown** | Preserves meaning | Destroys meaning |
| **Subword Validity** | All subwords valid | Most combinations invalid |
| **Vocabulary** | Open (can create new words) | Closed (6,985 fixed codes) |
| **BPE Alignment** | Aligns with morphemes | Random fragmentation |
| **Generation Strategy** | Subword composition | Whole-code selection |
| **Tokenization** | BPE works well | BPE catastrophic |

---

## Try It Yourself

### Exercise 1: Tokenize Medical Codes

```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

codes = ["401.9", "250.00", "428.0", "V58.61", "E849.0", "780.2"]

for code in codes:
    tokens = tokenizer.tokenize(code)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Code: {code:10} → Tokens: {tokens} (IDs: {token_ids})")
```

**Expected Output:**
```
Code: 401.9      → Tokens: ['Ġ401', '.', 'Ġ9'] (IDs: [20174, 4, 860])
Code: 250.00     → Tokens: ['Ġ250', '.', '00'] (IDs: [21268, 4, 713])
Code: 428.0      → Tokens: ['Ġ428', '.', '0'] (IDs: [34938, 4, 321])
Code: V58.61     → Tokens: ['ĠV', '58', '.', '61'] (IDs: [9732, 3943, 4, 4057])
Code: E849.0     → Tokens: ['ĠE', '849', '.', '0'] (IDs: [3608, 44293, 4, 321])
Code: 780.2      → Tokens: ['Ġ780', '.', '2'] (IDs: [9508, 4, 132])
```

**Observation:** Every single code is fragmented into 3-4 tokens!

### Exercise 2: Conceptual Question

**Question:** If you trained a model to generate phone numbers using BPE tokenization, what could go wrong?

**Hints:**
- Phone numbers: "555-1234", "555-5678"
- BPE might tokenize as: ["555", "-", "123", "4"], ["555", "-", "567", "8"]

**Answer:**
The model might generate:
- "555-123567" (wrong number of digits)
- "555-999999" (valid format but may not be a real number)
- "123-4" (incomplete)

Similarly, ICD-9 codes have specific format rules (XXX.XX) that BPE doesn't respect.

### Exercise 3: Identify Compositionality

**For each example, determine if meaning is compositional or atomic:**

1. "unbelievable" → ["un", "believe", "able"]
2. "555-1234" → ["555", "-", "1234"]
3. "401.9" → ["401", ".", "9"]
4. "COVID-19" → ["COVID", "-", "19"]
5. "http://example.com" → ["http", "://", "example", ".", "com"]

**Answers:**
1. **Compositional** ✓ (not + believe + capable = not capable of being believed)
2. **Atomic** ✗ (phone number is identifier, parts meaningless)
3. **Atomic** ✗ (medical code, parts meaningless)
4. **Atomic** ✗ (disease name, "19" doesn't mean 19th COVID)
5. **Mixed** ~ (protocol + domain parts have some meaning, but full URL is identifier)

---

## Key Takeaways

1. **Medical codes are atomic identifiers**, not compositional text. Breaking them apart destroys meaning.

2. **BPE tokenization fails for structured identifiers** because it treats them as arbitrary character sequences, not semantic units.

3. **Semantic integrity is critical:** ICD-9 codes must be tokenized as single tokens to preserve their medical meaning.

4. **Natural language intuitions mislead:** Techniques that work for text (BPE, subword composition) catastrophically fail for codes.

5. **The solution:** Custom tokenization with 1:1 code-to-token mapping (DiagnosisCodeTokenizer).

6. **Lesson from October 9, 2025:** Treating codes as text leads to complete generation failure (gibberish output).

---

## What's Next?

You now understand **why** medical codes require special handling compared to natural language.

**Next:** [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md) - See the catastrophic failure that occurred when we initially treated codes as text, and witness the "gibberish generation" that motivated the complete redesign.

**Alternative Path:**
- [01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md) - Review EHR basics if needed
- [04_SOLUTION_TOKEN_BASED_APPROACH.md](04_SOLUTION_TOKEN_BASED_APPROACH.md) - Skip ahead to the solution

---

**Navigation:**
- ← Back to [01_WHAT_IS_EHR.md](01_WHAT_IS_EHR.md)
- → Next: [03_THE_FRAGMENTATION_INCIDENT.md](03_THE_FRAGMENTATION_INCIDENT.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
