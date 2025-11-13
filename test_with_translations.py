"""Test script for patient generation with ICD-9 code translations."""
import logging
import sys
import torch
from pathlib import Path
from collections import Counter
import icd9cms

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_from_demographics


def get_code_description(code: str) -> str:
    """Get description for ICD-9 code using icd9cms package."""
    # Try exact match
    result = icd9cms.search(code)
    if result:
        # Result format: "code:short_desc:long_desc"
        parts = str(result).split(":")
        if len(parts) >= 3:
            return parts[2]  # Return long description
        elif len(parts) >= 2:
            return parts[1]  # Return short description if no long desc

    # Try with/without decimal
    if "." in code:
        code_nodot = code.replace(".", "")
        result = icd9cms.search(code_nodot)
        if result:
            parts = str(result).split(":")
            if len(parts) >= 3:
                return parts[2]
            elif len(parts) >= 2:
                return parts[1]
    else:
        # Try adding decimal in standard position
        if len(code) >= 3:
            code_withdot = f"{code[:3]}.{code[3:]}"
            result = icd9cms.search(code_withdot)
            if result:
                parts = str(result).split(":")
                if len(parts) >= 3:
                    return parts[2]
                elif len(parts) >= 2:
                    return parts[1]

    return None


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.Logger("test_with_translations")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def main():
    """Generate patients and display with code translations."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Patient Generation with ICD-9 Code Translations")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}\n")

    # Load vocabulary
    logger.info("Loading vocabulary...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Create tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}\n")

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

    # Generate patients
    logger.info("\n" + "=" * 80)
    logger.info("Generating 10 synthetic patients")
    logger.info("=" * 80)

    all_codes = []
    translated_count = 0
    total_count = 0

    for i in range(10):
        result = generate_patient_from_demographics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_sequence_length=256
        )

        # Display patient
        demographics = result['demographics']
        logger.info(f"\n{'█' * 80}")
        logger.info(f"PATIENT {i+1}")
        logger.info(f"{'█' * 80}")
        logger.info(f"Demographics: Age={demographics['age']:.1f}, Sex={demographics['sex']}")
        logger.info(f"Visits: {result['num_visits']}, Total codes: {result['num_codes']}\n")

        # Display visits with translations
        for visit_idx, visit_codes in enumerate(result['generated_visits']):
            logger.info(f"--- VISIT {visit_idx + 1} ({len(visit_codes)} codes) ---")

            for code_idx, code in enumerate(visit_codes, 1):
                description = get_code_description(code)
                all_codes.append(code)
                total_count += 1

                if description:
                    logger.info(f"  {code_idx:2d}. {code:10s} - {description}")
                    translated_count += 1
                else:
                    logger.info(f"  {code_idx:2d}. {code:10s} - [Unknown diagnosis]")

            logger.info("")

    # Statistics
    logger.info("=" * 80)
    logger.info("STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total codes generated: {total_count}")
    logger.info(f"Translated codes: {translated_count} ({translated_count/total_count*100:.1f}%)")
    logger.info(f"Unknown codes: {total_count - translated_count} ({(total_count-translated_count)/total_count*100:.1f}%)")

    # Most common codes
    code_counts = Counter(all_codes)
    logger.info(f"\nMost common codes generated:")
    for code, count in code_counts.most_common(20):
        description = get_code_description(code)
        if description:
            logger.info(f"  {code:10s} x{count:3d} - {description}")
        else:
            logger.info(f"  {code:10s} x{count:3d} - [Unknown]")

    logger.info("\n" + "=" * 80)
    logger.info("Generation complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
