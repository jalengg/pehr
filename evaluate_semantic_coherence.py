"""
Evaluate semantic coherence of generated patient sequences.

Measures clinical plausibility and distributional fidelity using:
1. Code frequency divergence (JS divergence)
2. Distribution match (KS tests)
3. Top-K code overlap
4. Pairwise co-occurrence score

Unlike Jaccard similarity, these metrics assess whether generated codes are
semantically plausible and statistically similar to training data, rather than
requiring exact code-level matches.
"""
import logging
import sys
import random
import torch
import numpy as np
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_sequence_conditional
from metrics import (
    compute_code_frequency_divergence,
    compute_distribution_match,
    compute_top_k_overlap,
    compute_cooccurrence_score
)


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("semantic_coherence")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def convert_result_to_patient_record(result: dict, subject_id: int):
    """Convert generation result to PatientRecord format for metrics."""
    from data_loader import PatientRecord

    # Create pseudo PatientRecord
    class GeneratedPatient:
        def __init__(self, visits, subject_id):
            self.visits = visits
            self.subject_id = subject_id

    return GeneratedPatient(result['generated_visits'], subject_id)


def main():
    """Main evaluation function."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Semantic Coherence Evaluation")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load training data
    logger.info("\nLoading MIMIC-III data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Split into train/test
    random.seed(config.training.seed)
    random.shuffle(patient_records)
    split_idx = int(len(patient_records) * (1 - config.data.train_val_split))
    train_patients = patient_records[:split_idx]
    test_patients = patient_records[split_idx:]

    logger.info(f"Training patients: {len(train_patients)}")
    logger.info(f"Test patients: {len(test_patients)}")

    # Create tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load trained model
    checkpoint_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model = load_trained_model(
        checkpoint_path=str(checkpoint_path),
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )

    # Generate patients (sample structure from test set, mask all codes)
    logger.info("\n" + "=" * 80)
    logger.info("Generating 100 patients (zero-prompt, sampled structure)")
    logger.info("=" * 80)

    generated_patients = []
    num_to_generate = min(100, len(test_patients))

    for i in range(num_to_generate):
        # Sample a random test patient for structure
        target_patient = test_patients[i]

        # Generate with complete masking (prompt_prob=0.0)
        result = generate_patient_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            target_patient=target_patient,
            device=device,
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            prompt_prob=0.0,  # Complete masking - no code prompts
            max_codes_per_visit=20
        )

        # Convert to PatientRecord format
        gen_patient = convert_result_to_patient_record(result, subject_id=f"gen_{i}")
        generated_patients.append(gen_patient)

        if (i + 1) % 20 == 0:
            logger.info(f"Generated {i + 1}/{num_to_generate} patients...")

    logger.info(f"Generation complete: {len(generated_patients)} patients")

    # Compute semantic coherence metrics
    logger.info("\n" + "=" * 80)
    logger.info("Computing Semantic Coherence Metrics")
    logger.info("=" * 80)

    # 1. Code Frequency Divergence
    logger.info("\n1. Code Frequency Distribution Match...")
    js_div = compute_code_frequency_divergence(generated_patients, train_patients)
    logger.info(f"   Jensen-Shannon Divergence: {js_div:.4f}")
    if js_div < 0.1:
        logger.info("   ✓ Excellent - distributions very similar")
    elif js_div < 0.3:
        logger.info("   ✓ Good - distributions reasonably similar")
    else:
        logger.info("   ✗ Poor - distributions differ significantly")

    # 2. Distribution Match Tests
    logger.info("\n2. Distribution Match Tests...")
    dist_match = compute_distribution_match(generated_patients, train_patients)
    logger.info(f"   Visits per patient: p={dist_match['visits_pvalue']:.4f}, stat={dist_match['visits_statistic']:.4f}")
    if dist_match['visits_pvalue'] > 0.05:
        logger.info("   ✓ Match - distributions are similar")
    else:
        logger.info("   ✗ Differ - distributions are different")

    logger.info(f"   Codes per visit: p={dist_match['codes_pvalue']:.4f}, stat={dist_match['codes_statistic']:.4f}")
    if dist_match['codes_pvalue'] > 0.05:
        logger.info("   ✓ Match - distributions are similar")
    else:
        logger.info("   ✗ Differ - distributions are different")

    # 3. Top-K Overlap
    logger.info("\n3. Top-100 Code Overlap...")
    top_k_overlap = compute_top_k_overlap(generated_patients, train_patients, k=100)
    logger.info(f"   Jaccard similarity: {top_k_overlap:.4f}")
    if top_k_overlap > 0.7:
        logger.info("   ✓ Excellent - learned most common codes")
    elif top_k_overlap > 0.5:
        logger.info("   ✓ Good - learned many common codes")
    else:
        logger.info("   ✗ Poor - missing common codes")

    # 4. Co-occurrence Score
    logger.info("\n4. Pairwise Co-occurrence Analysis...")
    cooccur_score = compute_cooccurrence_score(generated_patients, train_patients, logger)
    logger.info(f"   Average co-occurrence count: {cooccur_score:.2f}")
    if cooccur_score > 50:
        logger.info("   ✓ Excellent - codes frequently co-occur in training")
    elif cooccur_score > 20:
        logger.info("   ✓ Good - codes sometimes co-occur in training")
    else:
        logger.info("   ✗ Poor - codes rarely co-occur together")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Generated {len(generated_patients)} patients with zero-prompt conditioning")
    logger.info(f"Code Frequency Divergence: {js_div:.4f} ({'Good' if js_div < 0.3 else 'Poor'})")
    logger.info(f"Visit Distribution Match: p={dist_match['visits_pvalue']:.4f} ({'Match' if dist_match['visits_pvalue'] > 0.05 else 'Differ'})")
    logger.info(f"Codes Distribution Match: p={dist_match['codes_pvalue']:.4f} ({'Match' if dist_match['codes_pvalue'] > 0.05 else 'Differ'})")
    logger.info(f"Top-100 Overlap: {top_k_overlap:.4f} ({'Good' if top_k_overlap > 0.5 else 'Poor'})")
    logger.info(f"Co-occurrence Score: {cooccur_score:.2f} ({'Good' if cooccur_score > 20 else 'Poor'})")
    logger.info("\nInterpretation: Generated data is semantically {'coherent' if (js_div < 0.3 and top_k_overlap > 0.5 and cooccur_score > 20) else 'needs improvement'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
