"""
Generate synthetic patients using trained hierarchical model.
"""
import torch
import logging
import argparse
from pathlib import Path
from transformers import BartConfig
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from prompt_bart_model import PromptBartWithDemographicPrediction
from hierarchical_generation import generate_patient_hierarchical
import numpy as np


def load_hierarchical_model(checkpoint_path: str, tokenizer, config, device, logger):
    """Load trained hierarchical model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Initialize BART config
    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id
    # Keep default max_position_embeddings from BART (1024)

    # Initialize model
    model = PromptBartWithDemographicPrediction(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        d_hidden=config.model.d_hidden,
        prompt_length=config.model.prompt_length,
        age_loss_weight=config.model.age_loss_weight,
        sex_loss_weight=config.model.sex_loss_weight
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate patients with hierarchical model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_patients", type=int, default=100, help="Number of patients to generate")
    parser.add_argument("--output", type=str, default="generated_patients_hierarchical.txt", help="Output file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_categories", type=int, default=15, help="Max categories per patient")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Hierarchical Patient Generation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Num patients: {args.num_patients}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max categories: {args.max_categories}")

    # Load config
    config = Config.from_defaults()
    config.data.num_patients = 50000  # Match training (loads all 46,520 available patients)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load data and build hierarchy
    logger.info("\nLoading MIMIC-III data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    logger.info("Building ICD-9 hierarchy...")
    hierarchy = ICD9Hierarchy(vocab, logger=logger)
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"  Categories: {tokenizer.get_n_categories()}")
    logger.info(f"  Codes: {tokenizer.get_n_codes()}")

    # Load model
    logger.info("\nLoading trained model...")
    model = load_hierarchical_model(args.checkpoint, tokenizer, config, device, logger)

    # Sample demographics from training data
    ages = [p.age for p in patient_records]
    sexes = [p.gender_id for p in patient_records]

    # Generate patients
    logger.info(f"\nGenerating {args.num_patients} patients...")
    generated_patients = []

    for i in range(args.num_patients):
        # Sample demographics
        age = float(np.random.choice(ages))
        sex = int(np.random.choice(sexes))

        # Generate patient
        result = generate_patient_hierarchical(
            model=model,
            tokenizer=tokenizer,
            age=age,
            sex=sex,
            device=device,
            max_categories=args.max_categories,
            temperature=args.temperature,
            logger=None  # Suppress per-patient logging
        )

        generated_patients.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{args.num_patients} patients")

    # Save results
    logger.info(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        f.write("Generated Patients (Hierarchical Model)\n")
        f.write("=" * 80 + "\n\n")

        for i, patient in enumerate(generated_patients):
            f.write(f"Patient {i + 1}:\n")
            f.write(f"  Age: {patient['age']}\n")
            f.write(f"  Sex: {'M' if patient['sex'] == 0 else 'F'}\n")
            f.write(f"  Categories: {patient['n_categories']}\n")
            f.write(f"  Codes: {patient['n_codes']}\n")
            f.write(f"  Expansion ratio: {patient['expansion_ratio']:.2f}\n")
            f.write(f"  Codes: {', '.join(patient['codes'])}\n")
            f.write("\n")

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Generation Statistics")
    logger.info("=" * 80)

    avg_categories = np.mean([p['n_categories'] for p in generated_patients])
    avg_codes = np.mean([p['n_codes'] for p in generated_patients])
    avg_expansion = np.mean([p['expansion_ratio'] for p in generated_patients])

    logger.info(f"Average categories per patient: {avg_categories:.2f}")
    logger.info(f"Average codes per patient: {avg_codes:.2f}")
    logger.info(f"Average expansion ratio: {avg_expansion:.2f}")

    # Count unique codes
    all_codes = set()
    for p in generated_patients:
        all_codes.update(p['codes'])
    logger.info(f"Unique codes generated: {len(all_codes)}")

    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
