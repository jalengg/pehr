"""Test script for unconditional patient generation."""
import logging
import sys
import torch
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_from_demographics, sample_demographics


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

    # Test zero-prompt generation (demographic-conditional)
    logger.info("\n" + "=" * 80)
    logger.info("Generating 10 synthetic patients from demographics (zero code prompts)")
    logger.info("=" * 80)

    for i in range(10):
        logger.info(f"\n--- Patient {i+1} ---")

        # Generate from demographics only
        result = generate_patient_from_demographics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_sequence_length=256
        )

        # Print results
        demographics = result['demographics']
        logger.info(f"Demographics: Age={demographics['age']:.1f}, Sex={demographics['sex']}")
        logger.info(f"Generated {result['num_visits']} visits with {result['num_codes']} total codes")

        # Print visits
        for visit_idx, visit_codes in enumerate(result['generated_visits']):
            logger.info(f"  Visit {visit_idx + 1}: {', '.join(visit_codes[:10])}{'...' if len(visit_codes) > 10 else ''}")

    logger.info("\n" + "=" * 80)
    logger.info("Zero-prompt generation test complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
