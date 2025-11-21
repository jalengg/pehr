"""Test script for unconditional patient generation."""
import logging
import sys
import torch
from pathlib import Path
from collections import Counter

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_from_demographics, sample_demographics, generate_patient_warm_start, generate_patient_with_structure_constraints
from visit_structure_sampler import VisitStructureSampler


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.Logger("test_unconditional")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def main():
    """Test zero-prompt (demographic-conditional) generation."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Testing Zero-Prompt Patient Generation (Demographics Only)")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load vocabulary
    logger.info("\nLoading vocabulary...")
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

    # Initialize visit structure sampler from training data
    logger.info("\nInitializing visit structure sampler...")
    structure_sampler = VisitStructureSampler(patient_records, seed=42)
    logger.info(f"Visit structure sampler initialized:")
    logger.info(f"  {structure_sampler}")

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

    # Test zero-prompt generation (demographic-conditional) with structure constraints
    logger.info("\n" + "=" * 80)
    logger.info("Generating 100 synthetic patients with realistic visit structure constraints")
    logger.info("=" * 80)

    all_codes = []
    all_results = []

    for i in range(100):
        # Sample realistic visit structure
        target_structure = structure_sampler.sample_structure()

        # Generate patient with structure constraints
        result = generate_patient_with_structure_constraints(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_structure=target_structure,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )

        all_results.append(result)

        # Collect all codes
        for visit_codes in result['generated_visits']:
            all_codes.extend(visit_codes)

        # Print results for first 10 or every 10th patient
        if i < 10 or (i+1) % 10 == 0:
            logger.info(f"\n--- Patient {i+1} ---")
            demographics = result['demographics']
            logger.info(f"Demographics: Age={demographics['age']:.1f}, Sex={demographics['sex']}")
            logger.info(f"Generated {result['num_visits']} visits with {result['num_codes']} total codes")

            # Print visits
            for visit_idx, visit_codes in enumerate(result['generated_visits']):
                logger.info(f"  Visit {visit_idx + 1}: {', '.join(visit_codes[:10])}{'...' if len(visit_codes) > 10 else ''}")

    # Code diversity analysis
    logger.info("\n" + "=" * 80)
    logger.info("CODE DIVERSITY ANALYSIS")
    logger.info("=" * 80)

    unique_codes = set(all_codes)
    code_freq = Counter(all_codes)

    logger.info(f"\nTotal codes generated: {len(all_codes)}")
    logger.info(f"Unique codes: {len(unique_codes)}")
    logger.info(f"Vocabulary size: {len(vocab)} diagnosis codes")
    logger.info(f"Coverage: {len(unique_codes) / len(vocab) * 100:.2f}%")

    logger.info(f"\nTop 20 most frequent codes:")
    for code, count in code_freq.most_common(20):
        pct = count / len(all_codes) * 100
        logger.info(f"  {code}: {count} times ({pct:.1f}%)")

    logger.info(f"\nAll unique codes generated:")
    logger.info(f"  {sorted(unique_codes)}")

    logger.info("\n" + "=" * 80)
    logger.info("Structure-constrained generation test complete")
    logger.info("=" * 80)


def test_warm_start_generation():
    """Test warm start generation (PromptEHR paper's primary method)."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Testing Warm Start Patient Generation (50% Context)")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data and vocabulary
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
        return

    model = load_trained_model(
        checkpoint_path=str(checkpoint_path),
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )

    # Filter patients with at least 2 visits
    multi_visit_patients = [p for p in patient_records if len(p.visits) >= 2]
    logger.info(f"\nPatients with 2+ visits: {len(multi_visit_patients)} / {len(patient_records)}")

    if len(multi_visit_patients) == 0:
        logger.error("No patients with multiple visits found!")
        return

    # Test warm start generation at different context ratios
    context_ratios = [0.3, 0.5, 0.7]
    num_test_patients = 20

    for ratio in context_ratios:
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing Warm Start with {ratio*100:.0f}% Context")
        logger.info("=" * 80)

        results = []

        # Select random patients for testing
        import random
        test_patients = random.sample(multi_visit_patients, min(num_test_patients, len(multi_visit_patients)))

        for i, patient in enumerate(test_patients):
            logger.info(f"\n--- Patient {i+1}/{len(test_patients)} ---")
            logger.info(f"Total visits: {len(patient.visits)}")

            try:
                result = generate_patient_warm_start(
                    model=model,
                    tokenizer=tokenizer,
                    patient_record=patient,
                    context_ratio=ratio,
                    device=device,
                    max_length=512,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                )

                results.append(result)

                # Print results
                logger.info(f"  Context visits: {result['num_context_visits']}")
                logger.info(f"  Ground truth visits: {result['metrics']['num_gt_visits']}")
                logger.info(f"  Generated visits: {result['metrics']['num_gen_visits']}")
                logger.info(f"  Jaccard similarity: {result['metrics']['jaccard']:.3f}")
                logger.info(f"  Code overlap: {result['metrics']['overlap']}/{result['metrics']['gt_codes']} codes")

            except Exception as e:
                import traceback
                logger.error(f"  Error generating patient: {e}")
                traceback.print_exc()
                continue

        # Aggregate statistics
        if len(results) > 0:
            avg_jaccard = sum(r['metrics']['jaccard'] for r in results) / len(results)
            avg_overlap = sum(r['metrics']['overlap'] for r in results) / len(results)
            avg_gt_codes = sum(r['metrics']['gt_codes'] for r in results) / len(results)
            avg_gen_codes = sum(r['metrics']['gen_codes'] for r in results) / len(results)

            logger.info("\n" + "-" * 80)
            logger.info(f"SUMMARY (Context Ratio: {ratio*100:.0f}%)")
            logger.info("-" * 80)
            logger.info(f"Patients tested: {len(results)}")
            logger.info(f"Average Jaccard similarity: {avg_jaccard:.3f}")
            logger.info(f"Average code overlap: {avg_overlap:.1f}")
            logger.info(f"Average ground truth codes: {avg_gt_codes:.1f}")
            logger.info(f"Average generated codes: {avg_gen_codes:.1f}")

    logger.info("\n" + "=" * 80)
    logger.info("Warm start generation test complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--warm-start":
        test_warm_start_generation()
    else:
        main()
