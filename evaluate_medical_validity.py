"""
Evaluate medical validity of generated patient sequences.

Measures:
1. Duplicate code rate (should be ~0% with no_repeat_ngram_size=1)
2. Age-inappropriate codes (target: >98% appropriate)
3. Sex-inappropriate codes (target: >99% appropriate)
4. Jaccard similarity (target: 0.40-0.45)
"""
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_sequence_conditional


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("eval_validity")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# Age-inappropriate code rules from MEDICAL_VALIDITY.md
NEONATAL_ONLY_PREFIXES = ['V30', 'V31', '76', '77', 'V502', 'V290']

ADULT_MINIMUM_AGES = {
    '42979': 18,  # Atrial fibrillation - MOST COMMON ISSUE
    '41401': 30,  # Coronary atherosclerosis
    '412': 30,    # Old MI
    '4139': 30,   # Chronic ischemic heart disease
    '4240': 40,   # Mitral valve disorders
    '3572': 10,   # Diabetic polyneuropathy
    '2449': 10,   # Hypothyroidism
    '2720': 18,   # Pure hypercholesterolemia
    '2749': 18,   # Lipoid metabolism disorder
    '5855': 18,   # CKD Stage V
    '34690': 10,  # Migraine
}

# Sex-inappropriate codes
MALE_ONLY_CODES = ['V502']  # Circumcision
FEMALE_ONLY_PREFIXES = ['V27', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659']  # Pregnancy/childbirth


def check_duplicate_codes(visit: List[str]) -> Dict[str, int]:
    """Check for duplicate codes within a visit.

    Args:
        visit: List of ICD-9 codes for a single visit.

    Returns:
        Dictionary mapping duplicate codes to their counts.
    """
    code_counts = Counter(visit)
    duplicates = {code: count for code, count in code_counts.items() if count > 1}
    return duplicates


def is_age_appropriate(code: str, age: float) -> bool:
    """Check if code is age-appropriate.

    Args:
        code: ICD-9 diagnosis code.
        age: Patient age in years.

    Returns:
        True if code is appropriate for age, False otherwise.
    """
    # Rule 1: Neonatal-only codes (age 0-1)
    if age > 1:
        for prefix in NEONATAL_ONLY_PREFIXES:
            if code.startswith(prefix):
                return False

    # Rule 2: Adult minimum age requirements
    if code in ADULT_MINIMUM_AGES:
        if age < ADULT_MINIMUM_AGES[code]:
            return False

    return True


def is_sex_appropriate(code: str, sex: str) -> bool:
    """Check if code is sex-appropriate.

    Args:
        code: ICD-9 diagnosis code.
        sex: Patient sex ('M' or 'F').

    Returns:
        True if code is appropriate for sex, False otherwise.
    """
    # Male-only codes
    if sex == 'F' and code in MALE_ONLY_CODES:
        return False

    # Female-only codes (pregnancy/childbirth)
    if sex == 'M':
        for prefix in FEMALE_ONLY_PREFIXES:
            if code.startswith(prefix):
                return False

    return True


def compute_jaccard_similarity(generated: List[str], target: List[str]) -> float:
    """Compute Jaccard similarity between generated and target codes.

    Args:
        generated: Generated ICD-9 codes.
        target: Target ICD-9 codes.

    Returns:
        Jaccard similarity (intersection / union).
    """
    gen_set = set(generated)
    tgt_set = set(target)

    if len(gen_set) == 0 and len(tgt_set) == 0:
        return 1.0

    intersection = len(gen_set & tgt_set)
    union = len(gen_set | tgt_set)

    return intersection / union if union > 0 else 0.0


def evaluate_patient_generation(
    result: Dict,
    logger: logging.Logger
) -> Dict[str, float]:
    """Evaluate medical validity for a single patient generation.

    Args:
        result: Generation result from generate_patient_sequence_conditional()
                or generate_patient_from_demographics().
        logger: Logger instance.

    Returns:
        Dictionary with validation metrics.
    """
    generated_visits = result['generated_visits']
    target_visits = result.get('target_visits', None)  # None for unconditional
    demographics = result['demographics']

    age = demographics.get('age', None)
    sex = demographics.get('gender', demographics.get('sex', None))

    # Flatten all codes
    all_generated = [code for visit in generated_visits for code in visit]
    all_target = [code for visit in target_visits for code in visit] if target_visits else []

    metrics = {
        'total_codes': len(all_generated),
        'total_visits': len(generated_visits),
        'duplicate_codes': 0,
        'age_inappropriate': 0,
        'sex_inappropriate': 0,
        'jaccard_per_visit': [],
    }

    # Check each visit
    for visit_idx, gen_visit in enumerate(generated_visits):
        # Duplicate check
        duplicates = check_duplicate_codes(gen_visit)
        if duplicates:
            metrics['duplicate_codes'] += sum(count - 1 for count in duplicates.values())
            logger.warning(f"  Visit {visit_idx + 1} duplicates: {duplicates}")

        # Age appropriateness
        for code in gen_visit:
            if age is not None and not is_age_appropriate(code, age):
                metrics['age_inappropriate'] += 1
                logger.warning(f"  Visit {visit_idx + 1}: Age-inappropriate code '{code}' for age {age}")

        # Sex appropriateness
        for code in gen_visit:
            if sex is not None and not is_sex_appropriate(code, sex):
                metrics['sex_inappropriate'] += 1
                logger.warning(f"  Visit {visit_idx + 1}: Sex-inappropriate code '{code}' for sex {sex}")

        # Jaccard similarity (only if target visits available)
        if target_visits and visit_idx < len(target_visits):
            tgt_visit = target_visits[visit_idx]
            jaccard = compute_jaccard_similarity(gen_visit, tgt_visit)
            metrics['jaccard_per_visit'].append(jaccard)

    return metrics


def main():
    """Main evaluation function."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Medical Validity Evaluation")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load vocabulary
    logger.info("\nLoading MIMIC-III data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Create tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load trained model
    checkpoint_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, tokenizer, config, device, logger)

    # Split into train/val/test
    num_total = len(patient_records)
    num_train = int(num_total * 0.7)
    num_val = int(num_total * 0.15)
    test_patients = patient_records[num_train + num_val:]

    logger.info(f"\nTest set: {len(test_patients)} patients")

    # Evaluate on test patients
    num_evaluate = min(100, len(test_patients))  # Evaluate 100 test patients
    logger.info(f"Evaluating {num_evaluate} test patients...")

    # Aggregate metrics
    aggregate = {
        'total_codes': 0,
        'total_visits': 0,
        'duplicate_codes': 0,
        'age_inappropriate': 0,
        'sex_inappropriate': 0,
        'jaccard_scores': [],
    }

    for i, patient in enumerate(test_patients[:num_evaluate], 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{num_evaluate}")

        # Generate reconstruction
        result = generate_patient_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            target_patient=patient,
            device=device,
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            prompt_prob=0.5,
            max_codes_per_visit=20
        )

        # Evaluate
        metrics = evaluate_patient_generation(result, logger)

        # Aggregate
        aggregate['total_codes'] += metrics['total_codes']
        aggregate['total_visits'] += metrics['total_visits']
        aggregate['duplicate_codes'] += metrics['duplicate_codes']
        aggregate['age_inappropriate'] += metrics['age_inappropriate']
        aggregate['sex_inappropriate'] += metrics['sex_inappropriate']
        aggregate['jaccard_scores'].extend(metrics['jaccard_per_visit'])

    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nGeneration Statistics:")
    logger.info(f"  Patients evaluated: {num_evaluate}")
    logger.info(f"  Total visits: {aggregate['total_visits']}")
    logger.info(f"  Total codes: {aggregate['total_codes']}")
    logger.info(f"  Avg codes per visit: {aggregate['total_codes'] / aggregate['total_visits']:.2f}")

    logger.info(f"\nMedical Validity:")
    duplicate_rate = aggregate['duplicate_codes'] / aggregate['total_codes'] * 100 if aggregate['total_codes'] > 0 else 0
    logger.info(f"  Duplicate codes: {aggregate['duplicate_codes']} / {aggregate['total_codes']} ({duplicate_rate:.2f}%)")

    age_inappropriate_rate = aggregate['age_inappropriate'] / aggregate['total_codes'] * 100 if aggregate['total_codes'] > 0 else 0
    logger.info(f"  Age-inappropriate: {aggregate['age_inappropriate']} / {aggregate['total_codes']} ({age_inappropriate_rate:.2f}%)")

    sex_inappropriate_rate = aggregate['sex_inappropriate'] / aggregate['total_codes'] * 100 if aggregate['total_codes'] > 0 else 0
    logger.info(f"  Sex-inappropriate: {aggregate['sex_inappropriate']} / {aggregate['total_codes']} ({sex_inappropriate_rate:.2f}%)")

    logger.info(f"\nReconstruction Quality:")
    mean_jaccard = np.mean(aggregate['jaccard_scores']) if aggregate['jaccard_scores'] else 0.0
    std_jaccard = np.std(aggregate['jaccard_scores']) if aggregate['jaccard_scores'] else 0.0
    logger.info(f"  Mean Jaccard similarity: {mean_jaccard:.4f} ± {std_jaccard:.4f}")

    logger.info(f"\n" + "=" * 80)
    logger.info("Target Metrics:")
    logger.info(f"  Duplicate rate: <1% (got {duplicate_rate:.2f}%)")
    logger.info(f"  Age-appropriate: >98% (got {100 - age_inappropriate_rate:.2f}%)")
    logger.info(f"  Sex-appropriate: >99% (got {100 - sex_inappropriate_rate:.2f}%)")
    logger.info(f"  Jaccard similarity: 0.40-0.45 (got {mean_jaccard:.4f})")
    logger.info("=" * 80)

    # Pass/Fail summary
    logger.info("\nPass/Fail Summary:")
    passed = []
    failed = []

    if duplicate_rate < 1.0:
        passed.append("✓ Duplicate suppression")
    else:
        failed.append(f"✗ Duplicate suppression ({duplicate_rate:.2f}% > 1%)")

    if age_inappropriate_rate < 2.0:
        passed.append("✓ Age appropriateness")
    else:
        failed.append(f"✗ Age appropriateness ({age_inappropriate_rate:.2f}% > 2%)")

    if sex_inappropriate_rate < 1.0:
        passed.append("✓ Sex appropriateness")
    else:
        failed.append(f"✗ Sex appropriateness ({sex_inappropriate_rate:.2f}% > 1%)")

    if 0.40 <= mean_jaccard <= 0.50:
        passed.append("✓ Jaccard similarity")
    else:
        failed.append(f"✗ Jaccard similarity ({mean_jaccard:.4f} not in [0.40, 0.50])")

    for p in passed:
        logger.info(p)
    for f in failed:
        logger.info(f)

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
