"""
Data loading and preprocessing for MIMIC-III EHR data.
Separates demographics as conditioning variables from medical codes.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from vocabulary import DiagnosisVocabulary


# Ethnicity categories in MIMIC-III
ETHNICITY_CATEGORIES = [
    'WHITE',
    'BLACK',
    'HISPANIC OR LATINO',
    'ASIAN',
    'OTHER',
    'UNKNOWN/NOT SPECIFIED'
]

def normalize_ethnicity(ethnicity: str) -> str:
    """Normalize MIMIC-III ethnicity strings to canonical categories."""
    ethnicity = ethnicity.upper().strip()

    if 'WHITE' in ethnicity:
        return 'WHITE'
    elif 'BLACK' in ethnicity or 'AFRICAN' in ethnicity:
        return 'BLACK'
    elif 'HISPANIC' in ethnicity or 'LATINO' in ethnicity:
        return 'HISPANIC OR LATINO'
    elif 'ASIAN' in ethnicity:
        return 'ASIAN'
    elif 'UNKNOWN' in ethnicity or 'UNABLE' in ethnicity:
        return 'UNKNOWN/NOT SPECIFIED'
    else:
        return 'OTHER'


class PatientRecord:
    """Container for a single patient's EHR data."""

    def __init__(
        self,
        subject_id: int,
        age: float,
        gender: str,
        ethnicity: str,
        visits: List[List[str]]
    ):
        self.subject_id = subject_id
        self.age = age
        self.gender = gender
        self.ethnicity = ethnicity
        self.visits = visits  # List of lists: [[diag1, diag2], [diag3], ...]

    def to_dict(self) -> Dict:
        """Convert to dictionary format for dataset."""
        return {
            'subject_id': self.subject_id,
            'x_num': np.array([self.age], dtype=np.float32),
            'x_cat': np.array([
                1 if self.gender == 'F' else 0,  # 0=M, 1=F
                ETHNICITY_CATEGORIES.index(self.ethnicity)
            ], dtype=np.int64),
            'visits': self.visits,
            'num_visits': len(self.visits)
        }


def load_mimic_data(
    patients_path: str,
    admissions_path: str,
    diagnoses_path: str,
    logger: logging.Logger,
    num_patients: int = None
) -> Tuple[List[PatientRecord], DiagnosisVocabulary]:
    """Load MIMIC-III data and format into PatientRecord objects.

    Args:
        patients_path: Path to PATIENTS.csv file.
        admissions_path: Path to ADMISSIONS.csv file.
        diagnoses_path: Path to DIAGNOSES_ICD.csv file.
        logger: Logger instance for output.
        num_patients: Maximum number of patients to load.

    Returns:
        Tuple of (patient_records, diagnosis_vocabulary)
    """
    logger.info("Loading MIMIC-III data files")

    try:
        patients_df = pd.read_csv(patients_path, parse_dates=['DOB'])
        logger.info(f"Loaded {len(patients_df)} patients")

        admissions_df = pd.read_csv(admissions_path, parse_dates=['ADMITTIME'])
        logger.info(f"Loaded {len(admissions_df)} admissions")

        diagnoses_df = pd.read_csv(diagnoses_path)
        logger.info(f"Loaded {len(diagnoses_df)} diagnosis records")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e.filename}")
        return [], DiagnosisVocabulary()
    except Exception as e:
        logger.error(f"Unexpected error during file loading: {e}")
        return [], DiagnosisVocabulary()

    # Calculate age at first admission
    first_admissions = admissions_df.loc[
        admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
    ][['SUBJECT_ID', 'ADMITTIME']]

    demo_df = pd.merge(
        patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
        first_admissions,
        on='SUBJECT_ID',
        how='inner'
    )

    demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
    demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])

    # Merge with admissions to get ethnicity
    admissions_info = admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY']]

    # Merge admissions with diagnoses
    merged_df = pd.merge(
        admissions_info,
        diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    # Merge with demographics
    final_df = pd.merge(
        merged_df,
        demo_df[['SUBJECT_ID', 'AGE', 'GENDER']],
        on='SUBJECT_ID',
        how='left'
    )

    # Sort chronologically
    final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)

    logger.info("Processing patient records")

    # Build vocabulary and patient records
    vocab = DiagnosisVocabulary()
    patient_records = []

    patient_groups = final_df.groupby('SUBJECT_ID')

    for subject_id, patient_data in patient_groups:
        # Extract demographics
        age = float(patient_data['AGE'].iloc[0])
        gender = patient_data['GENDER'].iloc[0]

        # Normalize ethnicity to canonical categories
        raw_ethnicity = patient_data['ETHNICITY'].mode().iloc[0] if not patient_data['ETHNICITY'].mode().empty else 'UNKNOWN'
        ethnicity = normalize_ethnicity(raw_ethnicity)

        # Extract visits (grouped by HADM_ID)
        visits = []
        visit_groups = patient_data.groupby('HADM_ID', sort=False)

        for _, visit_data in visit_groups:
            # Get ICD-9 codes for this visit
            icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()

            # Add codes to vocabulary
            vocab.add_codes(icd_codes)

            visits.append(icd_codes)

        # Create patient record
        record = PatientRecord(
            subject_id=int(subject_id),
            age=age,
            gender=gender,
            ethnicity=ethnicity,
            visits=visits
        )
        patient_records.append(record)

        if num_patients is not None and len(patient_records) >= num_patients:
            break

    logger.info(f"Loaded {len(patient_records)} patient records")
    logger.info(f"Diagnosis vocabulary size: {len(vocab)}")

    # Log statistics
    if len(patient_records) > 0:
        avg_visits = np.mean([len(r.visits) for r in patient_records])
        avg_codes_per_visit = np.mean([len(code_list) for r in patient_records for code_list in r.visits])

        logger.info(f"Average visits per patient: {avg_visits:.2f}")
        logger.info(f"Average codes per visit: {avg_codes_per_visit:.2f}")

        # Gender distribution
        gender_counts = pd.Series([r.gender for r in patient_records]).value_counts()
        logger.info(f"Gender distribution: {gender_counts.to_dict()}")

        # Ethnicity distribution
        ethnicity_counts = pd.Series([r.ethnicity for r in patient_records]).value_counts()
        logger.info(f"Ethnicity distribution: {ethnicity_counts.to_dict()}")

        # Sample record
        sample = patient_records[0]
        logger.debug(f"Sample patient: age={sample.age}, gender={sample.gender}, ethnicity={sample.ethnicity}")
        logger.debug(f"Sample visits: {sample.visits[:2]}")  # First 2 visits

    return patient_records, vocab
