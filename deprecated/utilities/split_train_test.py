"""
Utility to split MIMIC-III patient records into train/test sets.
Saves split indices to enable reproducible generation experiments.
"""
import logging
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

from data_loader import load_mimic_data, PatientRecord


def setup_logging() -> logging.Logger:
    """Set up logging to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.Logger("split_data")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def split_train_test(
    patient_records: List[PatientRecord],
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    """Split patient records into train and test sets.

    Args:
        patient_records: List of all patient records.
        test_ratio: Fraction of data for test set (default: 0.2).
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (train_records, test_records).
    """
    np.random.seed(random_seed)

    num_patients = len(patient_records)
    num_test = int(num_patients * test_ratio)

    # Shuffle indices
    indices = np.random.permutation(num_patients)

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    train_records = [patient_records[i] for i in train_indices]
    test_records = [patient_records[i] for i in test_indices]

    return train_records, test_records


def main():
    """Load MIMIC-III data, split into train/test, and save."""
    logger = setup_logging()

    # Configuration
    PATIENTS_PATH = 'data_files/PATIENTS.csv'
    ADMISSIONS_PATH = 'data_files/ADMISSIONS.csv'
    DIAGNOSES_PATH = 'data_files/DIAGNOSES_ICD.csv'
    NUM_PATIENTS = 10000
    TEST_RATIO = 0.2
    RANDOM_SEED = 42
    OUTPUT_DIR = Path('data_splits')

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("Loading MIMIC-III data")
    patient_records, vocab = load_mimic_data(
        patients_path=PATIENTS_PATH,
        admissions_path=ADMISSIONS_PATH,
        diagnoses_path=DIAGNOSES_PATH,
        logger=logger,
        num_patients=NUM_PATIENTS
    )

    logger.info(f"Loaded {len(patient_records)} patients")
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Split train/test
    logger.info(f"Splitting data (test_ratio={TEST_RATIO})")
    train_records, test_records = split_train_test(
        patient_records,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )

    logger.info(f"Train set: {len(train_records)} patients")
    logger.info(f"Test set: {len(test_records)} patients")

    # Save splits
    train_path = OUTPUT_DIR / 'train_patients.pkl'
    test_path = OUTPUT_DIR / 'test_patients.pkl'
    vocab_path = OUTPUT_DIR / 'vocabulary.pkl'

    with open(train_path, 'wb') as f:
        pickle.dump(train_records, f)
    logger.info(f"Saved train set to {train_path}")

    with open(test_path, 'wb') as f:
        pickle.dump(test_records, f)
    logger.info(f"Saved test set to {test_path}")

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    logger.info(f"Saved vocabulary to {vocab_path}")

    # Log statistics
    logger.info("\n=== Train Set Statistics ===")
    train_visits = [len(r.visits) for r in train_records]
    train_codes = [len(code) for r in train_records for code in r.visits]
    logger.info(f"Average visits per patient: {np.mean(train_visits):.2f}")
    logger.info(f"Average codes per visit: {np.mean(train_codes):.2f}")

    logger.info("\n=== Test Set Statistics ===")
    test_visits = [len(r.visits) for r in test_records]
    test_codes = [len(code) for r in test_records for code in r.visits]
    logger.info(f"Average visits per patient: {np.mean(test_visits):.2f}")
    logger.info(f"Average codes per visit: {np.mean(test_codes):.2f}")


if __name__ == '__main__':
    main()
