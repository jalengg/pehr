"""
Generate synthetic patients via conditional reconstruction on test set.
Evaluates reconstruction quality using Jaccard similarity and exact match metrics.
"""
import logging
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

from config import Config
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_sequence_conditional
from data_loader import PatientRecord


def setup_logging(log_file: str = None) -> logging.Logger:
    """Set up logging to console and optionally to file.

    Args:
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("generate_test")
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets.

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0).
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def evaluate_reconstruction(
    generated_visits: List[List[str]],
    target_visits: List[List[str]]
) -> Dict[str, float]:
    """Evaluate reconstruction quality.

    Args:
        generated_visits: Generated visit sequences.
        target_visits: Original visit sequences.

    Returns:
        Dictionary with evaluation metrics:
            - jaccard_per_visit: List of Jaccard scores per visit
            - avg_jaccard: Average Jaccard across all visits
            - exact_match_rate: Fraction of visits with perfect reconstruction
            - avg_codes_generated: Average codes per generated visit
            - avg_codes_target: Average codes per target visit
    """
    jaccard_scores = []
    exact_matches = 0
    total_visits = len(target_visits)

    for gen_visit, target_visit in zip(generated_visits, target_visits):
        gen_set = set(gen_visit)
        target_set = set(target_visit)

        # Jaccard similarity
        jaccard = jaccard_similarity(gen_set, target_set)
        jaccard_scores.append(jaccard)

        # Exact match
        if gen_set == target_set:
            exact_matches += 1

    return {
        'jaccard_per_visit': jaccard_scores,
        'avg_jaccard': np.mean(jaccard_scores) if jaccard_scores else 0.0,
        'exact_match_rate': exact_matches / total_visits if total_visits > 0 else 0.0,
        'avg_codes_generated': np.mean([len(v) for v in generated_visits]) if generated_visits else 0.0,
        'avg_codes_target': np.mean([len(v) for v in target_visits]) if target_visits else 0.0,
        'num_visits': total_visits
    }


def main():
    """Run conditional reconstruction on test set and evaluate."""
    logger = setup_logging(log_file='generate_test.log')

    # Configuration
    CHECKPOINT_PATH = '/scratch/jalenj4/promptehr_checkpoints/best_model.pt'
    TEST_PATIENTS_PATH = 'data_splits/test_patients.pkl'
    VOCAB_PATH = 'data_splits/vocabulary.pkl'
    OUTPUT_DIR = Path('reconstruction_results')
    NUM_TEST_SAMPLES = 20
    TEMPERATURE = 0.3
    PROMPT_PROB = 0.5

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load vocabulary
    logger.info(f"Loading vocabulary from {VOCAB_PATH}")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # Initialize tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab=vocab)
    logger.info(f"Tokenizer initialized: {len(tokenizer)} tokens")

    # Load trained model
    model = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )

    # Load test patients
    logger.info(f"Loading test patients from {TEST_PATIENTS_PATH}")
    with open(TEST_PATIENTS_PATH, 'rb') as f:
        test_patients = pickle.load(f)

    logger.info(f"Loaded {len(test_patients)} test patients")

    # Sample test patients
    if len(test_patients) > NUM_TEST_SAMPLES:
        test_sample_indices = np.random.choice(len(test_patients), NUM_TEST_SAMPLES, replace=False)
        test_sample = [test_patients[i] for i in test_sample_indices]
    else:
        test_sample = test_patients

    logger.info(f"Evaluating on {len(test_sample)} patients")

    # Generate and evaluate
    all_metrics = []
    output_file = OUTPUT_DIR / 'reconstruction_results.txt'

    with open(output_file, 'w') as f:
        f.write("# Conditional Reconstruction Results\n")
        f.write(f"# Temperature: {TEMPERATURE}\n")
        f.write(f"# Prompt Probability: {PROMPT_PROB}\n")
        f.write(f"# Num Patients: {len(test_sample)}\n\n")

        for idx, patient in enumerate(test_sample):
            logger.info(f"Generating patient {idx + 1}/{len(test_sample)}")

            # Generate reconstruction
            result = generate_patient_sequence_conditional(
                model=model,
                tokenizer=tokenizer,
                target_patient=patient,
                device=device,
                temperature=TEMPERATURE,
                prompt_prob=PROMPT_PROB
            )

            # Evaluate
            metrics = evaluate_reconstruction(
                result['generated_visits'],
                result['target_visits']
            )
            all_metrics.append(metrics)

            # Write to file
            f.write(f"\n{'='*80}\n")
            f.write(f"Patient {idx + 1} (ID: {patient.subject_id})\n")
            f.write(f"Demographics: {result['demographics']}\n")
            f.write(f"Num Visits: {len(result['target_visits'])}\n")
            f.write(f"Avg Jaccard: {metrics['avg_jaccard']:.3f}\n")
            f.write(f"Exact Match Rate: {metrics['exact_match_rate']:.3f}\n\n")

            for visit_idx in range(len(result['target_visits'])):
                f.write(f"\n--- Visit {visit_idx + 1} ---\n")
                f.write(f"Target Codes ({len(result['target_visits'][visit_idx])}): {result['target_visits'][visit_idx]}\n")
                f.write(f"Prompt Codes ({len(result['prompt_codes'][visit_idx])}): {result['prompt_codes'][visit_idx]}\n")
                f.write(f"Generated Codes ({len(result['generated_visits'][visit_idx])}): {result['generated_visits'][visit_idx]}\n")
                f.write(f"Jaccard: {metrics['jaccard_per_visit'][visit_idx]:.3f}\n")

    # Aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE STATISTICS")
    logger.info("="*80)

    avg_jaccard = np.mean([m['avg_jaccard'] for m in all_metrics])
    avg_exact_match = np.mean([m['exact_match_rate'] for m in all_metrics])
    avg_codes_gen = np.mean([m['avg_codes_generated'] for m in all_metrics])
    avg_codes_target = np.mean([m['avg_codes_target'] for m in all_metrics])

    logger.info(f"Average Jaccard Similarity: {avg_jaccard:.3f}")
    logger.info(f"Average Exact Match Rate: {avg_exact_match:.3f}")
    logger.info(f"Average Codes Generated: {avg_codes_gen:.2f}")
    logger.info(f"Average Codes Target: {avg_codes_target:.2f}")

    # Save aggregate statistics
    stats_file = OUTPUT_DIR / 'aggregate_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("# Aggregate Statistics\n\n")
        f.write(f"Num Patients: {len(test_sample)}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Prompt Probability: {PROMPT_PROB}\n\n")
        f.write(f"Average Jaccard Similarity: {avg_jaccard:.3f}\n")
        f.write(f"Average Exact Match Rate: {avg_exact_match:.3f}\n")
        f.write(f"Average Codes Generated: {avg_codes_gen:.2f}\n")
        f.write(f"Average Codes Target: {avg_codes_target:.2f}\n")

    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Aggregate stats saved to {stats_file}")


if __name__ == '__main__':
    main()
