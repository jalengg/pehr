"""
PyTorch Dataset and DataCollator for EHR patient sequences.
Separates demographics (continuous conditioning) from diagnosis codes (discrete tokens).
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from data_loader import PatientRecord
from code_tokenizer import DiagnosisCodeTokenizer


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
        self.mask_token = tokenizer.MASK_TOKEN
        self.vocab_size = len(tokenizer.vocab)

    def mask_infill(
        self,
        visits: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[int]]]:
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
                [self.mask_token] +
                visit[start_idx + span_length:]
            )

            # Create label mask (1 for masked positions)
            label_mask = [0] * num_codes
            for i in range(start_idx, min(start_idx + span_length, num_codes)):
                label_mask[i] = 1

            corrupted_visits.append(corrupted_visit)
            label_masks.append(label_mask)

        return corrupted_visits, label_masks

    def del_token(self, visits: List[List[str]]) -> List[List[str]]:
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

    def rep_token(self, visits: List[List[str]]) -> List[List[str]]:
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
                    random_code = self.tokenizer.vocab.idx2code[random_code_indices[i]]
                    corrupted_visit.append(random_code)
                else:
                    corrupted_visit.append(code)

            corrupted_visits.append(corrupted_visit)

        return corrupted_visits


class EHRPatientDataset(Dataset):
    """PyTorch Dataset for patient EHR data with separated demographics and codes."""

    def __init__(
        self,
        patient_records: List[PatientRecord],
        tokenizer: DiagnosisCodeTokenizer,
        logger: logging.Logger
    ):
        """Initialize dataset.

        Args:
            patient_records: List of PatientRecord objects.
            tokenizer: DiagnosisCodeTokenizer instance.
            logger: Logger instance.
        """
        self.patient_records = patient_records
        self.tokenizer = tokenizer
        self.logger = logger

        if len(patient_records) > 0:
            sample = patient_records[0].to_dict()
            self.logger.debug(f"Sample x_num shape: {sample['x_num'].shape}")
            self.logger.debug(f"Sample x_cat shape: {sample['x_cat'].shape}")
            self.logger.debug(f"Sample num_visits: {sample['num_visits']}")

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single patient record as a dictionary.

        Returns dict with:
            - x_num: [1] array with age
            - x_cat: [2] array with [gender_id, ethnicity_id]
            - visit_codes: List[List[str]] of diagnosis codes
            - token_ids: [seq_len] encoded visit sequence
        """
        record = self.patient_records[idx]
        record_dict = record.to_dict()

        # Encode visits to token IDs
        token_ids = self.tokenizer.encode_patient(record.visits, add_special_tokens=True)

        return {
            'x_num': record_dict['x_num'],
            'x_cat': record_dict['x_cat'],
            'visit_codes': record.visits,
            'token_ids': np.array(token_ids, dtype=np.int64),
            'subject_id': record.subject_id
        }


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

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
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
                        task_type='mask_infilling'
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
        item: Dict,
        visits: List[List[str]],
        task_type: str
    ) -> Dict:
        """Create a single training sample from visits.

        Args:
            item: Original patient item from dataset.
            visits: List of visits (possibly corrupted).
            task_type: String identifier for debugging.

        Returns:
            Dictionary with x_num, x_cat, token_ids, task_type.
        """
        # Shuffle code order within each visit to treat codes as unordered sets
        # This matches PromptEHR's approach and prevents positional bias
        shuffled_visits = []
        for visit in visits:
            if len(visit) > 0:
                # random.sample shuffles the list
                shuffled_visit = list(np.random.choice(visit, len(visit), replace=False))
            else:
                shuffled_visit = []
            shuffled_visits.append(shuffled_visit)

        # Encode visits to token IDs
        token_ids = self.tokenizer.encode_patient(shuffled_visits, add_special_tokens=True)

        return {
            'x_num': item['x_num'],
            'x_cat': item['x_cat'],
            'token_ids': np.array(token_ids, dtype=np.int64),
            'task_type': task_type
        }

    def _create_next_visit_sample(self, item: Dict) -> Dict:
        """Create next-visit prediction sample (TPL task).

        Randomly select split point N, create:
        - Input: demographics + visits[0:N] + <mask>
        - Labels: full sequence including visit N+1

        Args:
            item: Original patient item from dataset.

        Returns:
            Dictionary with input sequence for next-visit prediction.
        """
        visits = item['visit_codes']
        num_visits = len(visits)

        # Randomly select split point (predict visit N from visits 0:N-1)
        split_idx = np.random.randint(0, num_visits - 1)

        # Input: visits up to split_idx + <mask> placeholder
        input_visits = visits[:split_idx] + [[self.tokenizer.MASK_TOKEN]]

        # Encode input with mask
        token_ids = self.tokenizer.encode_patient(input_visits, add_special_tokens=False)

        # Add actual next visit for labels (model learns to predict this)
        next_visit_ids = self.tokenizer.encode_visit(visits[split_idx], add_markers=True)
        token_ids.extend(next_visit_ids)

        # Add BOS and END tokens
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.convert_tokens_to_ids('<END>')]

        return {
            'x_num': item['x_num'],
            'x_cat': item['x_cat'],
            'token_ids': np.array(token_ids, dtype=np.int64),
            'task_type': 'next_visit_prediction'
        }

    def _collate_samples(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch multiple samples with padding.

        Args:
            samples: List of sample dictionaries.

        Returns:
            Batched tensors.
        """
        # Stack demographic features
        x_num = torch.stack([torch.from_numpy(s['x_num']) for s in samples])
        x_cat = torch.stack([torch.from_numpy(s['x_cat']) for s in samples])

        # Pad token sequences
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

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

        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list)
        }
