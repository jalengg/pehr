"""
Hierarchical PyTorch Dataset for category-based EHR training.

This dataset trains the model on ICD-9 category sequences instead of specific codes,
leveraging the 8x coverage improvement from hierarchical generation.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import numpy as np
import logging
from data_loader import PatientRecord
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer


class HierarchicalEHRDataset(Dataset):
    """PyTorch Dataset for hierarchical (category-based) EHR training.

    This dataset encodes patient records as category sequences for training.
    During generation, categories are expanded to specific codes using sampling.
    """

    def __init__(
        self,
        patient_records: List[PatientRecord],
        tokenizer: HierarchicalDiagnosisTokenizer,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize hierarchical dataset.

        Args:
            patient_records: List of PatientRecord objects.
            tokenizer: HierarchicalDiagnosisTokenizer instance.
            logger: Optional logger instance.
        """
        self.patient_records = patient_records
        self.tokenizer = tokenizer
        self.logger = logger

        if logger and len(patient_records) > 0:
            sample = patient_records[0].to_dict()
            logger.debug(f"Sample x_num shape: {sample['x_num'].shape}")
            logger.debug(f"Sample x_cat shape: {sample['x_cat'].shape}")
            logger.debug(f"Sample num_visits: {sample['num_visits']}")

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single patient record as category sequence.

        Returns dict with:
            - x_num: [1] array with age
            - x_cat: [2] array with [gender_id, ethnicity_id]
            - visit_codes: List[List[str]] of original diagnosis codes
            - token_ids: [seq_len] encoded category sequence
            - subject_id: Patient ID string
        """
        record = self.patient_records[idx]
        record_dict = record.to_dict()

        # Encode visits as category sequence
        token_ids = self.tokenizer.encode_patient_as_categories(record.visits)

        return {
            'x_num': record_dict['x_num'],
            'x_cat': record_dict['x_cat'],
            'visit_codes': record.visits,  # Keep original codes for reference
            'token_ids': np.array(token_ids, dtype=np.int64),
            'subject_id': record.subject_id
        }


class HierarchicalEHRDataCollator:
    """Data collator for hierarchical EHR sequences with category-level corruption.

    Implements data corruption strategies on category sequences:
    1. Mask infilling: Replace category spans with <mask> token
    2. Token deletion: Randomly delete categories
    3. Token replacement: Replace categories with random alternatives
    4. Next visit prediction: Predict next visit from previous visits
    """

    def __init__(
        self,
        tokenizer: HierarchicalDiagnosisTokenizer,
        max_seq_length: int = 512,
        logger: Optional[logging.Logger] = None,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15,
        corruption_prob: float = 0.5,
        use_mask_infilling: bool = True,
        use_token_deletion: bool = False,
        use_token_replacement: bool = False,
        use_next_visit_prediction: bool = False
    ):
        """Initialize data collator.

        Args:
            tokenizer: HierarchicalDiagnosisTokenizer instance.
            max_seq_length: Maximum sequence length.
            logger: Optional logger.
            lambda_poisson: Poisson lambda for span masking length.
            del_probability: Probability of deleting each token.
            rep_probability: Probability of replacing each token.
            corruption_prob: Probability of applying corruption to a sample.
            use_mask_infilling: Enable mask infilling corruption.
            use_token_deletion: Enable token deletion corruption.
            use_token_replacement: Enable token replacement corruption.
            use_next_visit_prediction: Enable next visit prediction task.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.logger = logger
        self.lambda_poisson = lambda_poisson
        self.del_probability = del_probability
        self.rep_probability = rep_probability
        self.corruption_prob = corruption_prob
        self.use_mask_infilling = use_mask_infilling
        self.use_token_deletion = use_token_deletion
        self.use_token_replacement = use_token_replacement
        self.use_next_visit_prediction = use_next_visit_prediction

    def apply_mask_infilling(self, token_ids: List[int]) -> tuple[List[int], List[int]]:
        """Apply Poisson span masking to category tokens.

        Args:
            token_ids: List of token IDs (including special tokens).

        Returns:
            Tuple of (corrupted_token_ids, label_mask).
        """
        # Extract category token positions (skip special tokens)
        category_positions = []
        for i, token_id in enumerate(token_ids):
            if self.tokenizer.is_category_token(token_id):
                category_positions.append(i)

        if len(category_positions) == 0:
            return token_ids, [0] * len(token_ids)

        # Sample span length from Poisson distribution
        span_length = max(1, min(len(category_positions),
                                np.random.poisson(self.lambda_poisson)))

        # Randomly select start position within category tokens
        max_start = len(category_positions) - span_length
        start_idx = np.random.randint(0, max(1, max_start + 1))

        # Get actual token positions to mask
        mask_start_pos = category_positions[start_idx]
        mask_end_pos = category_positions[min(start_idx + span_length - 1, len(category_positions) - 1)]

        # Create corrupted sequence
        corrupted = (
            token_ids[:mask_start_pos] +
            [self.tokenizer.mask_token_id] +
            token_ids[mask_end_pos + 1:]
        )

        # Create label mask (1 for positions that should be predicted)
        label_mask = [0] * len(token_ids)
        for pos in range(mask_start_pos, mask_end_pos + 1):
            label_mask[pos] = 1

        return corrupted, label_mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of patient records.

        Args:
            batch: List of patient record dicts from dataset.

        Returns:
            Dictionary with batched tensors:
            - input_ids: [batch_size, max_len] padded token sequences
            - attention_mask: [batch_size, max_len] attention mask
            - labels: [batch_size, max_len] labels for language modeling
            - x_num: [batch_size, 1] continuous age values
            - x_cat: [batch_size, 2] categorical gender/ethnicity IDs
        """
        batch_size = len(batch)

        # Extract demographics
        x_num = np.stack([sample['x_num'] for sample in batch])
        x_cat = np.stack([sample['x_cat'] for sample in batch])

        # Process token sequences
        input_ids_list = []
        labels_list = []

        for sample in batch:
            token_ids = sample['token_ids'].tolist()

            # Standard autoregressive training (no corruption for hierarchical training)
            # The key benefit comes from category-level learning, not data augmentation
            input_ids_list.append(token_ids)
            labels_list.append(token_ids.copy())

        # Pad sequences
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_seq_length)

        input_ids = np.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        labels = np.full((batch_size, max_len), -100, dtype=np.int64)

        for i, (input_seq, label_seq) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = min(len(input_seq), max_len)

            input_ids[i, :seq_len] = input_seq[:seq_len]
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = label_seq[:seq_len]

            # Mask padding tokens in labels
            for j in range(seq_len):
                if input_ids[i, j] == self.tokenizer.pad_token_id:
                    labels[i, j] = -100

        return {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'labels': torch.from_numpy(labels),
            'x_num': torch.from_numpy(x_num).float(),
            'x_cat': torch.from_numpy(x_cat).long()
        }
