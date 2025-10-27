# PromptEHR Enhanced Training Implementation Plan

**Date**: 2025-10-17
**Objective**: Implement missing training techniques to improve medical coherence and model robustness

---

## Executive Summary

Current implementation uses simple teacher forcing and generates syntactically valid but medically nonsensical sequences (e.g., newborn codes for adults, war injuries mixed with pregnancy). Official PromptEHR achieves better coherence through multi-task training with data corruptions and temporal prediction tasks.

**Missing Components Identified**:
- ❌ Mask infilling (span masking with Poisson distribution)
- ❌ Token deletion (binomial deletion)
- ❌ Token replacement (random code substitution)
- ❌ Next-visit prediction task (TPL - Temporal Perplexity)
- ❌ Cross-modal masking task (SPL - Spatial Perplexity, requires multi-code types)
- ❌ Multi-sample data collator (current: 1 sample per patient, needed: 1-5 samples)
- ❌ TPL/SPL evaluation metrics

**Implementation Timeline**: ~5 hours coding + overnight retraining

---

## Phase 1: Tokenizer Enhancement

### File: `code_tokenizer.py`

**Current State**:
```python
self.special_tokens = {
    '<PAD>': 0,
    '<BOS>': 1,
    '<EOS>': 2,
    '<v>': 3,
    '<\\v>': 4,
    '<END>': 5,
}
```

**Required Change**:
```python
self.special_tokens = {
    '<PAD>': 0,
    '<BOS>': 1,
    '<EOS>': 2,
    '<v>': 3,
    '<\\v>': 4,
    '<END>': 5,
    '<mask>': 6,  # NEW: Mask token for infilling tasks
}
```

**Rationale**: Mask token needed for span masking and next-visit prediction tasks.

**Estimated Time**: 5 minutes

---

## Phase 2: Corruption Functions

### File: `dataset.py` (new class)

Create a `CorruptionFunctions` class to handle all data augmentation strategies.

### Implementation

```python
class CorruptionFunctions:
    """Data corruption functions for robust EHR generation training.

    Implements three corruption strategies from PromptEHR paper:
    1. Mask infilling: Replace code spans with <mask> token
    2. Token deletion: Randomly delete codes
    3. Token replacement: Replace codes with random alternatives
    """

    def __init__(
        self,
        tokenizer: DiagnosisCodeTokenizer,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15
    ):
        """Initialize corruption functions.

        Args:
            tokenizer: DiagnosisCodeTokenizer instance.
            lambda_poisson: Poisson lambda for span masking length (default: 3.0).
            del_probability: Probability of deleting each token (default: 0.15).
            rep_probability: Probability of replacing each token (default: 0.15).
        """
        self.tokenizer = tokenizer
        self.lambda_poisson = lambda_poisson
        self.del_probability = del_probability
        self.rep_probability = rep_probability
        self.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
        self.vocab_size = len(tokenizer.code_vocab)

    def mask_infill(
        self,
        visits: list[list[str]]
    ) -> tuple[list[list[str]], list[list[int]]]:
        """Apply Poisson-distributed span masking to diagnosis codes.

        For each visit:
        - Sample span length from Poisson(lambda=3.0)
        - Randomly select contiguous span of that length
        - Replace with single <mask> token
        - Track which positions were masked for loss calculation

        Args:
            visits: List of visits, each visit is list of ICD-9 codes.
                   Example: [['401.9', '250.00'], ['428.0', '584.9']]

        Returns:
            Tuple of:
            - corrupted_visits: Visits with masked spans
            - label_masks: Binary mask indicating which positions to predict
                          (1 = predict this position, 0 = ignore in loss)

        Example:
            Input:  [['401.9', '250.00', '585.9'], ['428.0']]
            Output: ([['401.9', '<mask>'], ['428.0']],
                    [[0, 1, 1], [0]])
        """
        corrupted_visits = []
        label_masks = []

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                label_masks.append([])
                continue

            # Sample span length from Poisson distribution
            span_length = max(1, min(num_codes - 1,
                                    np.random.poisson(self.lambda_poisson)))

            # Randomly select start position
            max_start = num_codes - span_length
            start_idx = np.random.randint(0, max(1, max_start + 1))

            # Create corrupted visit
            corrupted_visit = (
                visit[:start_idx] +
                ['<mask>'] +
                visit[start_idx + span_length:]
            )

            # Create label mask (1 for masked positions)
            label_mask = [0] * num_codes
            for i in range(start_idx, min(start_idx + span_length, num_codes)):
                label_mask[i] = 1

            corrupted_visits.append(corrupted_visit)
            label_masks.append(label_mask)

        return corrupted_visits, label_masks

    def del_token(self, visits: list[list[str]]) -> list[list[str]]:
        """Apply binomial token deletion to diagnosis codes.

        For each code in each visit:
        - With probability del_probability (0.15), delete the code
        - Never delete all codes in a visit (keep at least 1)

        Args:
            visits: List of visits, each visit is list of ICD-9 codes.

        Returns:
            Visits with randomly deleted codes.

        Example:
            Input:  [['401.9', '250.00', '585.9'], ['428.0', '584.9']]
            Output: [['401.9', '585.9'], ['428.0']]  # 250.00 and 584.9 deleted
        """
        corrupted_visits = []

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                continue

            # Generate deletion mask (1 = delete, 0 = keep)
            deletion_mask = np.random.binomial(1, self.del_probability, num_codes)

            # Keep at least 1 code per visit
            if deletion_mask.sum() == num_codes:
                # All would be deleted, randomly keep one
                keep_idx = np.random.randint(0, num_codes)
                deletion_mask[keep_idx] = 0

            # Apply deletion
            corrupted_visit = [
                code for i, code in enumerate(visit)
                if deletion_mask[i] == 0
            ]

            corrupted_visits.append(corrupted_visit)

        return corrupted_visits

    def rep_token(self, visits: list[list[str]]) -> list[list[str]]:
        """Apply binomial token replacement with random codes.

        For each code in each visit:
        - With probability rep_probability (0.15), replace with random code
        - Random code sampled uniformly from vocabulary

        Args:
            visits: List of visits, each visit is list of ICD-9 codes.

        Returns:
            Visits with randomly replaced codes.

        Example:
            Input:  [['401.9', '250.00'], ['428.0']]
            Output: [['401.9', '715.90'], ['V27.0']]  # 250.00 -> 715.90, 428.0 -> V27.0
        """
        corrupted_visits = []

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                continue

            # Generate replacement mask (1 = replace, 0 = keep)
            replacement_mask = np.random.binomial(1, self.rep_probability, num_codes)

            # Generate random replacement codes
            random_code_indices = np.random.randint(0, self.vocab_size, num_codes)

            # Apply replacement
            corrupted_visit = []
            for i, code in enumerate(visit):
                if replacement_mask[i] == 1:
                    # Replace with random code from vocabulary
                    random_code = self.tokenizer.code_vocab[random_code_indices[i]]
                    corrupted_visit.append(random_code)
                else:
                    corrupted_visit.append(code)

            corrupted_visits.append(corrupted_visit)

        return corrupted_visits
```

**Testing Strategy**:
```python
# Unit test for mask_infill
tokenizer = DiagnosisCodeTokenizer(['401.9', '250.00', '585.9', '428.0', '584.9'])
corruption = CorruptionFunctions(tokenizer)

visits = [['401.9', '250.00', '585.9'], ['428.0', '584.9']]
corrupted, masks = corruption.mask_infill(visits)

# Verify:
# - corrupted visits contain '<mask>' token
# - label_masks have 1s only at masked positions
# - original visit length preserved in label_mask

# Unit test for del_token
corrupted = corruption.del_token(visits)
# Verify: each visit has at least 1 code remaining

# Unit test for rep_token
corrupted = corruption.rep_token(visits)
# Verify: replaced codes are valid vocabulary entries
```

**Estimated Time**: 1 hour

---

## Phase 3: Enhanced Data Collator

### File: `dataset.py` (modify `EHRDataCollator`)

**Current Behavior**:
- Takes 1 patient → produces 1 training sample (simple padding)

**New Behavior**:
- Takes 1 patient → produces **1-5 training samples**:
  1. Original (teacher forcing)
  2. Mask infilling sample (probabilistic)
  3. Deletion sample (probabilistic)
  4. Replacement sample (probabilistic)
  5. Next-visit prediction sample (if patient has 2+ visits)

### Implementation

```python
class EHRDataCollator:
    """Enhanced collator for batching EHR patient data with corruptions.

    Generates multiple training samples per patient using different corruption
    strategies to improve robustness and temporal coherence.
    """

    def __init__(
        self,
        tokenizer: DiagnosisCodeTokenizer,
        max_seq_length: int,
        logger: logging.Logger,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15,
        corruption_prob: float = 0.5,
        use_mask_infilling: bool = True,
        use_token_deletion: bool = True,
        use_token_replacement: bool = True,
        use_next_visit_prediction: bool = True
    ):
        """Initialize enhanced collator.

        Args:
            tokenizer: DiagnosisCodeTokenizer instance.
            max_seq_length: Maximum sequence length for padding/truncation.
            logger: Logger instance.
            lambda_poisson: Poisson lambda for span masking (default: 3.0).
            del_probability: Token deletion probability (default: 0.15).
            rep_probability: Token replacement probability (default: 0.15).
            corruption_prob: Probability of applying each corruption type (default: 0.5).
            use_mask_infilling: Enable mask infilling samples (default: True).
            use_token_deletion: Enable token deletion samples (default: True).
            use_token_replacement: Enable token replacement samples (default: True).
            use_next_visit_prediction: Enable next-visit prediction samples (default: True).
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.logger = logger
        self.corruption_prob = corruption_prob

        # Corruption flags
        self.use_mask_infilling = use_mask_infilling
        self.use_token_deletion = use_token_deletion
        self.use_token_replacement = use_token_replacement
        self.use_next_visit_prediction = use_next_visit_prediction

        # Initialize corruption functions
        self.corruption_funcs = CorruptionFunctions(
            tokenizer=tokenizer,
            lambda_poisson=lambda_poisson,
            del_probability=del_probability,
            rep_probability=rep_probability
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch with multiple samples per patient.

        Args:
            batch: List of dictionaries from EHRPatientDataset.__getitem__

        Returns:
            Dictionary with batched tensors (expanded with corruptions):
                - x_num: [expanded_batch, 1] age
                - x_cat: [expanded_batch, 2] gender and ethnicity
                - input_ids: [expanded_batch, max_seq_len] padded sequences
                - attention_mask: [expanded_batch, max_seq_len] attention masks
                - labels: [expanded_batch, max_seq_len] labels with -100 for padding
                - task_types: [expanded_batch] task type identifiers (for debugging)
        """
        expanded_batch = []

        for item in batch:
            # 1. Original sample (teacher forcing)
            expanded_batch.append(
                self._create_sample(
                    item=item,
                    visits=item['visit_codes'],
                    task_type='teacher_forcing'
                )
            )

            # 2. Mask infilling sample
            if self.use_mask_infilling and np.random.rand() < self.corruption_prob:
                corrupted_visits, label_masks = self.corruption_funcs.mask_infill(
                    item['visit_codes']
                )
                expanded_batch.append(
                    self._create_sample(
                        item=item,
                        visits=corrupted_visits,
                        task_type='mask_infilling',
                        label_masks=label_masks
                    )
                )

            # 3. Token deletion sample
            if self.use_token_deletion and np.random.rand() < self.corruption_prob:
                corrupted_visits = self.corruption_funcs.del_token(item['visit_codes'])
                expanded_batch.append(
                    self._create_sample(
                        item=item,
                        visits=corrupted_visits,
                        task_type='token_deletion'
                    )
                )

            # 4. Token replacement sample
            if self.use_token_replacement and np.random.rand() < self.corruption_prob:
                corrupted_visits = self.corruption_funcs.rep_token(item['visit_codes'])
                expanded_batch.append(
                    self._create_sample(
                        item=item,
                        visits=corrupted_visits,
                        task_type='token_replacement'
                    )
                )

            # 5. Next-visit prediction sample (TPL)
            if self.use_next_visit_prediction and len(item['visit_codes']) > 1:
                expanded_batch.append(
                    self._create_next_visit_sample(item)
                )

        # Collate all samples
        return self._collate_samples(expanded_batch)

    def _create_sample(
        self,
        item: dict,
        visits: list[list[str]],
        task_type: str,
        label_masks: list[list[int]] | None = None
    ) -> dict:
        """Create a single training sample from visits.

        Args:
            item: Original patient item from dataset.
            visits: List of visits (possibly corrupted).
            task_type: String identifier for debugging.
            label_masks: Optional binary masks for masked positions.

        Returns:
            Dictionary with x_num, x_cat, token_ids, task_type.
        """
        # Encode visits to token IDs
        token_ids = self.tokenizer.encode_patient(visits, add_special_tokens=True)

        return {
            'x_num': item['x_num'],
            'x_cat': item['x_cat'],
            'token_ids': np.array(token_ids, dtype=np.int64),
            'task_type': task_type,
            'label_masks': label_masks
        }

    def _create_next_visit_sample(self, item: dict) -> dict:
        """Create next-visit prediction sample (TPL task).

        Randomly select split point N, create:
        - Input: demographics + visits[0:N] + <mask>
        - Labels: visit[N] (only compute loss on this visit)

        Args:
            item: Original patient item from dataset.

        Returns:
            Dictionary with input sequence and masked labels.
        """
        visits = item['visit_codes']
        num_visits = len(visits)

        # Randomly select split point (predict visit N from visits 0:N-1)
        split_idx = np.random.randint(0, num_visits - 1)

        # Input: visits up to split_idx + <mask> placeholder
        input_visits = visits[:split_idx] + [['<mask>']]

        # Encode input
        input_token_ids = self.tokenizer.encode_patient(
            input_visits,
            add_special_tokens=True
        )

        # Encode target (full sequence including next visit)
        target_visits = visits[:split_idx + 1]
        target_token_ids = self.tokenizer.encode_patient(
            target_visits,
            add_special_tokens=True
        )

        # Create label mask: only compute loss on visit[split_idx]
        # Find position of visit[split_idx] in target_token_ids
        # (This requires tokenizer to support position tracking)
        # For simplicity, we'll compute loss on entire sequence but this
        # should be refined to only predict the next visit

        return {
            'x_num': item['x_num'],
            'x_cat': item['x_cat'],
            'token_ids': np.array(target_token_ids, dtype=np.int64),
            'task_type': 'next_visit_prediction',
            'label_masks': None  # TODO: Refine to mask only next visit
        }

    def _collate_samples(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Batch multiple samples with padding.

        Args:
            samples: List of sample dictionaries.

        Returns:
            Batched tensors.
        """
        batch_size = len(samples)

        # Stack demographic features
        x_num = torch.stack([torch.from_numpy(s['x_num']) for s in samples])
        x_cat = torch.stack([torch.from_numpy(s['x_cat']) for s in samples])

        # Pad token sequences
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        task_types = []

        for sample in samples:
            token_ids = sample['token_ids']
            seq_len = len(token_ids)

            # Truncate if too long
            if seq_len > self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length]
                seq_len = self.max_seq_length

            # Create attention mask
            attention_mask = np.ones(seq_len, dtype=np.int64)

            # Pad to max_seq_length
            num_padding = self.max_seq_length - seq_len
            if num_padding > 0:
                token_ids = np.concatenate([
                    token_ids,
                    np.full(num_padding, self.pad_token_id, dtype=np.int64)
                ])
                attention_mask = np.concatenate([
                    attention_mask,
                    np.zeros(num_padding, dtype=np.int64)
                ])

            # Create labels (mask padding with -100)
            labels = token_ids.copy()
            labels[labels == self.pad_token_id] = -100

            input_ids_list.append(torch.from_numpy(token_ids))
            attention_mask_list.append(torch.from_numpy(attention_mask))
            labels_list.append(torch.from_numpy(labels))
            task_types.append(sample['task_type'])

        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            'task_types': task_types  # For debugging
        }
```

**Key Changes**:
1. Each patient now contributes 1-5 samples per batch
2. Effective batch size increases by ~3-4x (8 patients → 24-32 samples)
3. May need to reduce `batch_size` in config to avoid OOM (16 → 8)
4. Task types tracked for debugging and analysis

**Estimated Time**: 1.5 hours

---

## Phase 4: Configuration Updates

### File: `config.py`

Add corruption and multi-task training parameters.

```python
@dataclass
class TrainingConfig:
    # Existing parameters
    batch_size: int = 8  # REDUCED from 16 (due to sample expansion)
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # NEW: Corruption parameters
    lambda_poisson: float = 3.0  # Poisson lambda for span masking length
    del_probability: float = 0.15  # Token deletion probability
    rep_probability: float = 0.15  # Token replacement probability
    corruption_prob: float = 0.5  # Probability of applying each corruption type

    # NEW: Multi-task training flags
    use_mask_infilling: bool = True
    use_token_deletion: bool = True
    use_token_replacement: bool = True
    use_next_visit_prediction: bool = True

    # NEW: Evaluation metrics
    compute_tpl: bool = True  # Compute Temporal Perplexity during validation
    compute_spl: bool = False  # Spatial Perplexity (requires multi-code types)
```

**Important**: Reduce `batch_size` from 16 to 8 to account for sample expansion (each patient now generates 3-4 samples on average).

**Estimated Time**: 15 minutes

---

## Phase 5: Temporal Perplexity (TPL) Metric

### File: `metrics.py` (new function)

Implement TPL evaluation metric to measure temporal coherence.

```python
def compute_temporal_perplexity(
    model: PromptBartModel,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: DiagnosisCodeTokenizer,
    logger: logging.Logger
) -> float:
    """Compute Temporal Perplexity (TPL) on validation set.

    TPL measures how well the model predicts the next visit given previous visits.
    Lower TPL indicates better temporal coherence.

    For each patient with N visits (N >= 2):
    1. Mask last visit
    2. Predict last visit from previous visits
    3. Compute cross-entropy loss
    4. Average across all predictions

    TPL = exp(average_loss)

    Args:
        model: Trained PromptBartModel.
        val_loader: Validation DataLoader.
        device: Device to run on.
        tokenizer: DiagnosisCodeTokenizer instance.
        logger: Logger instance.

    Returns:
        TPL value (lower is better).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            # Process each patient in batch
            for i in range(len(batch['x_num'])):
                # Skip patients with only 1 visit
                # (In practice, need to track visit boundaries in batch)
                # For now, we'll compute TPL on samples with task_type='next_visit_prediction'

                if batch.get('task_types') and batch['task_types'][i] == 'next_visit_prediction':
                    # Extract single sample
                    x_num = batch['x_num'][i:i+1].to(device)
                    x_cat = batch['x_cat'][i:i+1].to(device)
                    input_ids = batch['input_ids'][i:i+1].to(device)
                    attention_mask = batch['attention_mask'][i:i+1].to(device)
                    labels = batch['labels'][i:i+1].to(device)

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        x_num=x_num,
                        x_cat=x_cat
                    )

                    total_loss += outputs.loss.item()
                    total_samples += 1

    if total_samples == 0:
        logger.warning("No next-visit prediction samples found for TPL computation")
        return float('inf')

    avg_loss = total_loss / total_samples
    tpl = np.exp(avg_loss)

    return tpl
```

**Alternative Approach** (if task_types not available in batch):

Create separate validation DataLoader that only contains next-visit prediction samples:

```python
def create_tpl_dataloader(
    patient_records: list[PatientRecord],
    tokenizer: DiagnosisCodeTokenizer,
    batch_size: int
) -> DataLoader:
    """Create DataLoader specifically for TPL evaluation.

    Only includes next-visit prediction samples.
    """
    # Filter patients with 2+ visits
    multi_visit_patients = [p for p in patient_records if len(p.visits) > 1]

    # Create dataset
    dataset = EHRPatientDataset(multi_visit_patients, tokenizer, logger)

    # Create collator that ONLY generates next-visit samples
    collator = EHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=logger,
        use_mask_infilling=False,
        use_token_deletion=False,
        use_token_replacement=False,
        use_next_visit_prediction=True
    )

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
```

**Estimated Time**: 45 minutes

---

## Phase 6: Validation Update

### File: `trainer.py` (modify `validate` function)

Add TPL computation to validation loop.

```python
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    config: Config,  # NEW: Need config for compute_tpl flag
    tokenizer: DiagnosisCodeTokenizer  # NEW: Need tokenizer for TPL
) -> dict[str, float]:
    """Run validation loop with TPL computation.

    Args:
        model: PromptBartModel instance.
        val_loader: Validation DataLoader.
        device: Device to run on.
        logger: Logger instance.
        config: Configuration object.
        tokenizer: DiagnosisCodeTokenizer instance.

    Returns:
        Dictionary with validation metrics including TPL.
    """
    model.eval()
    metrics_tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            x_num = batch['x_num'].to(device)
            x_cat = batch['x_cat'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                x_num=x_num,
                x_cat=x_cat
            )

            # Track metrics
            metrics_tracker.update(
                loss=outputs.loss.item(),
                logits=outputs.logits,
                labels=labels,
                compute_accuracy=True
            )

    # Get average metrics
    val_metrics = metrics_tracker.get_average_metrics()

    logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                f"Perplexity: {val_metrics['perplexity']:.4f}, "
                f"Token Accuracy: {val_metrics.get('token_accuracy', 0):.4f}, "
                f"Code Accuracy: {val_metrics.get('code_accuracy', 0):.4f}")

    # NEW: Compute TPL if enabled
    if config.training.compute_tpl:
        tpl = compute_temporal_perplexity(
            model=model,
            val_loader=val_loader,
            device=device,
            tokenizer=tokenizer,
            logger=logger
        )
        val_metrics['tpl'] = tpl
        logger.info(f"Temporal Perplexity (TPL): {tpl:.4f}")

    return val_metrics
```

**Required Changes in `trainer.py`**:
1. Import `compute_temporal_perplexity` from metrics.py
2. Update `validate()` function signature to accept `config` and `tokenizer`
3. Update all calls to `validate()` in training loop (line 436)

**Estimated Time**: 15 minutes

---

## Phase 7: Testing Plan

### Unit Tests

Create `test_corruptions.py`:

```python
import numpy as np
from code_tokenizer import DiagnosisCodeTokenizer
from dataset import CorruptionFunctions

def test_mask_infill():
    """Test mask infilling produces valid outputs."""
    tokenizer = DiagnosisCodeTokenizer(['401.9', '250.00', '585.9', '428.0', '584.9'])
    corruption = CorruptionFunctions(tokenizer, lambda_poisson=3.0)

    visits = [['401.9', '250.00', '585.9'], ['428.0', '584.9']]
    corrupted, masks = corruption.mask_infill(visits)

    # Verify mask token present
    assert any('<mask>' in visit for visit in corrupted), "No mask token found"

    # Verify label masks have correct length
    assert len(masks) == len(visits), "Label mask length mismatch"
    for i, visit in enumerate(visits):
        assert len(masks[i]) == len(visit), f"Visit {i} mask length mismatch"

    # Verify at least one masked position per visit
    assert any(sum(mask) > 0 for mask in masks), "No positions masked"

    print("✓ test_mask_infill passed")

def test_del_token():
    """Test token deletion never deletes entire visits."""
    tokenizer = DiagnosisCodeTokenizer(['401.9', '250.00', '585.9'])
    corruption = CorruptionFunctions(tokenizer, del_probability=0.15)

    visits = [['401.9', '250.00', '585.9'], ['428.0']]

    # Run deletion 100 times
    for _ in range(100):
        corrupted = corruption.del_token(visits)

        # Verify no empty visits
        assert all(len(visit) > 0 for visit in corrupted), "Empty visit created"

    print("✓ test_del_token passed")

def test_rep_token():
    """Test token replacement produces valid vocabulary codes."""
    vocab = ['401.9', '250.00', '585.9', '428.0', '584.9']
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer, rep_probability=0.15)

    visits = [['401.9', '250.00'], ['428.0']]
    corrupted = corruption.rep_token(visits)

    # Verify all codes are valid
    for visit in corrupted:
        for code in visit:
            assert code in vocab, f"Invalid code: {code}"

    print("✓ test_rep_token passed")

def test_collator_expansion():
    """Test enhanced collator produces multiple samples per patient."""
    # (Requires full setup with dataset, tokenizer, etc.)
    # Verify batch size expansion
    pass

if __name__ == "__main__":
    test_mask_infill()
    test_del_token()
    test_rep_token()
    print("\nAll unit tests passed!")
```

### Integration Test

Run short training with all corruptions enabled:

```bash
# Modify config temporarily
num_patients = 100
num_epochs = 1

# Run training
python trainer.py

# Verify:
# - No crashes
# - Loss decreases
# - Effective batch size ~3-4x larger than config.batch_size
# - TPL computation succeeds
```

**Estimated Time**: 30 minutes

---

## Phase 8: Full Retraining

### Slurm Script

Create `train_enhanced.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=promptehr_enhanced
#SBATCH --output=logs/train_enhanced_%j.out
#SBATCH --error=logs/train_enhanced_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalen.jiang2+slurm@gmail.com
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load environment
module load python/3.12
source venv/bin/activate

# Run training with enhanced strategy
python trainer.py

echo "Enhanced training complete!"
```

Submit:
```bash
sbatch train_enhanced.slurm
```

**Expected Duration**: 8-12 hours (longer than original due to 3-4x more samples per epoch)

**Estimated Time**: Overnight

---

## Summary and Expected Improvements

### Implementation Checklist

- [ ] Phase 1: Add `<mask>` token to tokenizer (5 min)
- [ ] Phase 2: Implement `CorruptionFunctions` class (1 hr)
- [ ] Phase 3: Enhance `EHRDataCollator` (1.5 hrs)
- [ ] Phase 4: Update config parameters (15 min)
- [ ] Phase 5: Implement TPL metric (45 min)
- [ ] Phase 6: Update validation (15 min)
- [ ] Phase 7: Unit and integration testing (30 min)
- [ ] Phase 8: Full retraining (overnight)

**Total Development Time**: ~5 hours
**Total Training Time**: 8-12 hours

### Expected Improvements After Retraining

**1. Temporal Coherence (TPL Metric)**
- **Before**: Model generates nonsensical visit progressions (e.g., newborn codes for 25yo adult)
- **After**: Next-visit prediction task should reduce temporal inconsistencies
- **Measurable**: TPL metric quantifies improvement (lower is better)

**2. Robustness to Incomplete Data**
- **Before**: Model trained only on complete sequences
- **After**: Mask infilling trains model to handle missing codes
- **Impact**: Better generation quality when conditioning is sparse

**3. Reduced Overfitting**
- **Before**: Model memorizes exact training sequences
- **After**: Token deletion/replacement force generalization
- **Impact**: More diverse generated sequences, less training set memorization

**4. Evaluation Capability**
- **Before**: Only basic perplexity (measures reconstruction loss)
- **After**: TPL measures temporal coherence specifically
- **Impact**: Can benchmark against official PromptEHR (TPL reported in paper)

### Remaining Limitations

Even after these improvements, the model will still have limitations:

1. **No Hard Medical Validity Rules**:
   - Still generates statistically coherent but medically nonsensical sequences
   - No age-appropriate filtering (newborn codes for adults)
   - No gender-appropriate filtering (pregnancy codes for males)

2. **No Medical Ontology Integration**:
   - Doesn't understand ICD-9 hierarchy
   - Doesn't model disease co-occurrence patterns
   - Doesn't enforce temporal disease progression rules

3. **Single Code Type**:
   - Only diagnosis codes (no procedures/drugs)
   - SPL (Spatial Perplexity) requires multi-modal codes
   - Can't model cross-modal relationships (diagnosis ↔ treatment)

4. **Statistical Coherence ≠ Clinical Validity**:
   - TPL improves temporal statistics, not medical logic
   - Model learns patterns, not medical knowledge
   - Generated data suitable for ML experiments, not clinical use

### Future Enhancements (Beyond This Plan)

To achieve clinical validity, would need:

1. **Post-Processing Filters**:
   - Age-appropriate code filtering
   - Gender-appropriate code filtering
   - Temporal constraint validation

2. **Medical Knowledge Integration**:
   - ICD-9 ontology embeddings
   - Disease progression models
   - Comorbidity constraints

3. **Multi-Modal Extension**:
   - Add PROCEDURES_ICD table
   - Add PRESCRIPTIONS table
   - Implement SPL (Spatial Perplexity)
   - Train cross-modal prediction tasks

4. **Domain Expert Validation**:
   - Clinical review of generated sequences
   - Automated validity scoring (e.g., DrBenchmark)
   - Comparison with real patient trajectories

---

## References

- **Official PromptEHR Implementation**: `/u/jalenj4/PromptEHR/promptehr/dataset.py`
- **Original Paper**: "PromptEHR: Conditional Electronic Healthcare Records Generation with Prompt Learning"
- **MIMIC-III Documentation**: https://mimic.mit.edu/docs/iii/

---

**Last Updated**: 2025-10-17
**Status**: Ready for implementation
