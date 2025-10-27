"""
PyTorch Dataset and DataCollator for EHR patient sequences.
Separates demographics (continuous conditioning) from diagnosis codes (discrete tokens).
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import numpy as np
import logging
from data_loader import PatientRecord
from code_tokenizer import DiagnosisCodeTokenizer


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
    """Collator for batching EHR patient data with padding."""

    def __init__(
        self,
        tokenizer: DiagnosisCodeTokenizer,
        max_seq_length: int,
        logger: logging.Logger
    ):
        """Initialize collator.

        Args:
            tokenizer: DiagnosisCodeTokenizer instance.
            max_seq_length: Maximum sequence length (for padding/truncation).
            logger: Logger instance.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.logger = logger

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of patient records.

        Args:
            batch: List of dictionaries from EHRPatientDataset.__getitem__

        Returns:
            Dictionary with batched tensors:
                - x_num: [batch, 1] age
                - x_cat: [batch, 2] gender and ethnicity
                - input_ids: [batch, max_seq_len] padded token sequences
                - attention_mask: [batch, max_seq_len] 1 for real tokens, 0 for padding
                - labels: [batch, max_seq_len] same as input_ids but with -100 for padding
        """
        batch_size = len(batch)

        # Stack demographic features
        x_num = torch.stack([torch.from_numpy(item['x_num']) for item in batch])
        x_cat = torch.stack([torch.from_numpy(item['x_cat']) for item in batch])

        # Pad token sequences
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            token_ids = item['token_ids']
            seq_len = len(token_ids)

            # Truncate if too long
            if seq_len > self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length]
                seq_len = self.max_seq_length

            # Create attention mask (1 for real tokens)
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

        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)

        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
