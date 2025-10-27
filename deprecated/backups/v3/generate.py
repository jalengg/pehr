"""
Generate synthetic patient sequences using trained PromptEHR model.
"""
import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from transformers import BartConfig

from config import Config
from data_loader import load_mimic_data, PatientRecord
from code_tokenizer import DiagnosisCodeTokenizer
from prompt_bart_model import PromptBartModel


def setup_logging() -> logging.Logger:
    """Set up logging to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("generator")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_trained_model(
    checkpoint_path: str,
    tokenizer: DiagnosisCodeTokenizer,
    config: Config,
    device: torch.device,
    logger: logging.Logger
) -> PromptBartModel:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        tokenizer: DiagnosisCodeTokenizer instance.
        config: Configuration object.
        device: Device to load model on.
        logger: Logger instance.

    Returns:
        Loaded PromptBartModel instance.
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Initialize model architecture
    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id

    model = PromptBartModel(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        d_hidden=config.model.d_hidden,
        prompt_length=config.model.prompt_length
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model


def generate_patient_sequence_conditional(
    model: PromptBartModel,
    tokenizer: DiagnosisCodeTokenizer,
    target_patient: 'PatientRecord',
    device: torch.device,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    prompt_prob: float = 0.5,
    max_codes_per_visit: int = 20
) -> dict:
    """Generate synthetic patient via conditional reconstruction (PromptEHR approach).

    Given a real patient from test set, randomly masks ~50% of codes and reconstructs
    the full visit structure. This matches original PromptEHR's evaluation protocol.

    Args:
        model: Trained PromptBartModel.
        tokenizer: DiagnosisCodeTokenizer instance.
        target_patient: PatientRecord from test set to reconstruct.
        device: Device to run on.
        temperature: Sampling temperature (default: 0.3).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        prompt_prob: Probability of keeping each code as prompt (default: 0.5).
        max_codes_per_visit: Cap visit codes at this number (default: 20).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of generated code sequences
            - 'target_visits': List[List[str]] of original codes
            - 'prompt_codes': List[List[str]] of codes provided as prompts
            - 'demographics': dict of patient demographics
    """
    model.eval()

    # Extract demographics
    age = target_patient.age
    gender = 1 if target_patient.gender == 'F' else 0
    ethnicity_map = {
        'WHITE': 0, 'BLACK': 1, 'HISPANIC OR LATINO': 2,
        'ASIAN': 3, 'OTHER': 4, 'UNKNOWN/NOT SPECIFIED': 5
    }
    ethnicity = ethnicity_map.get(target_patient.ethnicity, 5)

    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[gender, ethnicity]], dtype=torch.long).to(device)

    # Initialize accumulators
    generated_visits = []
    prompt_codes_per_visit = []

    # Create dummy encoder input (prompts are in decoder)
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    # Initialize decoder input with BOS
    input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        # Process each visit from target patient
        for visit_idx, target_codes in enumerate(target_patient.visits):
            # Step 1: Cap codes at max_codes_per_visit
            num_codes = len(target_codes)
            if num_codes > max_codes_per_visit:
                target_codes = list(np.random.choice(target_codes, max_codes_per_visit, replace=False))
                num_codes = max_codes_per_visit

            if num_codes == 0:
                # Empty visit - skip
                generated_visits.append([])
                prompt_codes_per_visit.append([])
                continue

            # Step 2: Randomly mask ~50% of codes (binomial sampling)
            keep_mask = np.random.binomial(1, prompt_prob, num_codes).astype(bool)
            prompt_codes = [code for i, code in enumerate(target_codes) if keep_mask[i]]

            # Step 3: Encode prompt codes
            v_token_id = tokenizer.convert_tokens_to_ids("<v>")
            v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")

            prompt_token_ids = [v_token_id]
            for code in prompt_codes:
                # Codes are in vocab, need to add code_offset to get token ID
                code_idx = tokenizer.vocab.code2idx[code]
                code_token_id = tokenizer.code_offset + code_idx
                prompt_token_ids.append(code_token_id)

            # Append prompt to input_ids
            prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, prompt_tensor], dim=1)

            # Step 4: Generate to reconstruct full visit
            max_new_tokens = num_codes + 2  # Target length

            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=input_ids,
                    x_num=x_num,
                    x_cat=x_cat
                )

                # Get next token logits
                next_token_logits = outputs.logits[0, -1, :] / temperature

                # Suppress special tokens
                next_token_logits[tokenizer.bos_token_id] = float('-inf')
                next_token_logits[tokenizer.eos_token_id] = float('-inf')
                next_token_logits[tokenizer.pad_token_id] = float('-inf')

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Append to input
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

                # Stop at </v>
                if next_token_id == v_end_token_id:
                    break

            # Step 5: Extract generated codes
            visit_token_ids = input_ids[0, -len(prompt_token_ids) - max_new_tokens:].cpu().tolist()
            generated_code_ids = [
                tid for tid in visit_token_ids
                if tid >= tokenizer.code_offset and tid not in [v_token_id, v_end_token_id]
            ]

            # Decode codes (convert token IDs back to diagnosis codes)
            generated_codes = []
            for tid in generated_code_ids:
                code_idx = tid - tokenizer.code_offset
                code = tokenizer.vocab.idx2code[code_idx]
                generated_codes.append(code)

            # Step 6: Combine with prompt codes and deduplicate
            all_codes = list(set(generated_codes + prompt_codes))

            # Ensure exactly num_codes by sampling if needed
            if len(all_codes) < num_codes:
                # Not enough unique codes generated - resample with replacement
                needed = num_codes - len(all_codes)
                additional = list(np.random.choice(generated_codes, needed, replace=True)) if len(generated_codes) > 0 else []
                all_codes.extend(additional)
            elif len(all_codes) > num_codes:
                # Too many codes - sample exactly num_codes
                all_codes = list(np.random.choice(all_codes, num_codes, replace=False))

            generated_visits.append(all_codes)
            prompt_codes_per_visit.append(prompt_codes)

    return {
        'generated_visits': generated_visits,
        'target_visits': target_patient.visits,
        'prompt_codes': prompt_codes_per_visit,
        'demographics': {
            'age': age,
            'gender': target_patient.gender,
            'ethnicity': target_patient.ethnicity
        }
    }


def decode_patient_demographics(age: float, gender: int, ethnicity: int) -> dict[str, str]:
    """Decode demographics back to readable format.

    Args:
        age: Normalized age value.
        gender: Gender category index.
        ethnicity: Ethnicity category index.

    Returns:
        Dictionary with decoded demographics.
    """
    # Gender mapping (from data_loader.py)
    gender_map = {0: "F", 1: "M"}

    # Ethnicity mapping (from data_loader.py)
    ethnicity_map = {
        0: "WHITE",
        1: "BLACK",
        2: "HISPANIC",
        3: "ASIAN",
        4: "OTHER",
        5: "UNKNOWN"
    }

    return {
        "age": f"{age:.1f}",
        "gender": gender_map.get(gender, "UNKNOWN"),
        "ethnicity": ethnicity_map.get(ethnicity, "UNKNOWN")
    }


def main():
    """Main generation function."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PromptEHR Synthetic Patient Generation")
    logger.info("=" * 80)

    # Load configuration
    config = Config.from_defaults()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load vocabulary (must match training: num_patients=3000)
    logger.info("\nLoading vocabulary...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients  # Use same as training (3000)
    )

    # Create tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load trained model
    # Check both old and new checkpoint locations
    checkpoint_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        # Fallback to old location
        checkpoint_path = Path("checkpoints") / "best_model.pt"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found in either location")
        logger.error(f"  - {config.training.checkpoint_dir}/best_model.pt")
        logger.error(f"  - checkpoints/best_model.pt")
        logger.error("Please train the model first using trainer.py")
        sys.exit(1)

    logger.info(f"Using checkpoint: {checkpoint_path}")

    model = load_trained_model(checkpoint_path, tokenizer, config, device, logger)

    # Generate synthetic patients
    logger.info("\n" + "=" * 80)
    logger.info("Generating Synthetic Patients")
    logger.info("=" * 80)

    # Define diverse demographics for generation (20 patients)
    test_demographics = [
        {"age": 65.0, "gender": 1, "ethnicity": 0},  # 65yo white male
        {"age": 45.0, "gender": 0, "ethnicity": 1},  # 45yo black female
        {"age": 30.0, "gender": 1, "ethnicity": 2},  # 30yo hispanic male
        {"age": 70.0, "gender": 0, "ethnicity": 0},  # 70yo white female
        {"age": 55.0, "gender": 1, "ethnicity": 3},  # 55yo asian male
        {"age": 80.0, "gender": 0, "ethnicity": 0},  # 80yo white female
        {"age": 25.0, "gender": 0, "ethnicity": 2},  # 25yo hispanic female
        {"age": 60.0, "gender": 1, "ethnicity": 1},  # 60yo black male
        {"age": 50.0, "gender": 0, "ethnicity": 4},  # 50yo other female
        {"age": 75.0, "gender": 1, "ethnicity": 0},  # 75yo white male
        {"age": 40.0, "gender": 0, "ethnicity": 0},  # 40yo white female
        {"age": 85.0, "gender": 1, "ethnicity": 0},  # 85yo white male
        {"age": 35.0, "gender": 1, "ethnicity": 1},  # 35yo black male
        {"age": 72.0, "gender": 0, "ethnicity": 3},  # 72yo asian female
        {"age": 28.0, "gender": 0, "ethnicity": 2},  # 28yo hispanic female
        {"age": 58.0, "gender": 1, "ethnicity": 0},  # 58yo white male
        {"age": 48.0, "gender": 0, "ethnicity": 1},  # 48yo black female
        {"age": 90.0, "gender": 0, "ethnicity": 0},  # 90yo white female
        {"age": 62.0, "gender": 1, "ethnicity": 4},  # 62yo other male
        {"age": 42.0, "gender": 0, "ethnicity": 3},  # 42yo asian female
    ]

    for i, demo in enumerate(test_demographics, 1):
        logger.info(f"\n{'─' * 80}")

        # Decode demographics
        decoded_demo = decode_patient_demographics(
            demo["age"], demo["gender"], demo["ethnicity"]
        )

        logger.info(f"Patient {i}:")
        logger.info(f"  Demographics: {decoded_demo['age']}yo {decoded_demo['ethnicity']} {decoded_demo['gender']}")

        # Generate sequence with lower temperature to reduce inappropriate codes
        sequence = generate_patient_sequence(
            model=model,
            tokenizer=tokenizer,
            age=demo["age"],
            gender=demo["gender"],
            ethnicity=demo["ethnicity"],
            device=device,
            max_length=50,  # Realistic based on training data
            temperature=0.3,  # LOWERED from 0.8 to stick to high-probability codes
            top_k=40,
            top_p=0.9,
            min_codes_per_visit=3,  # Minimum codes before allowing visit end
            max_codes_per_visit=15  # Maximum codes per visit
        )

        logger.info(f"  Generated Sequence:")
        logger.info(f"    {sequence}")

        # Count visits and codes
        num_visits = sequence.count("<v>")
        codes = [token for token in sequence.split() if token not in ["<s>", "</s>", "<v>", "<\v>"]]
        num_codes = len(codes)

        logger.info(f"  Statistics:")
        logger.info(f"    Visits: {num_visits}")
        logger.info(f"    Diagnosis Codes: {num_codes}")
        logger.info(f"    Avg Codes per Visit: {num_codes / num_visits if num_visits > 0 else 0:.1f}")

    logger.info("\n" + "=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
