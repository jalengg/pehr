"""
Generate synthetic patient sequences using trained PromptEHR model.
"""
import logging
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from transformers import BartConfig

from config import Config
from data_loader import load_mimic_data
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


def generate_patient_sequence(
    model: PromptBartModel,
    tokenizer: DiagnosisCodeTokenizer,
    age: float,
    gender: int,
    ethnicity: int,
    device: torch.device,
    max_length: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    min_codes_per_visit: int = 3,
    max_codes_per_visit: int = 20
) -> str:
    """Generate a synthetic patient sequence with visit structure enforcement.

    Args:
        model: Trained PromptBartModel.
        tokenizer: DiagnosisCodeTokenizer instance.
        age: Patient age (0-90).
        gender: Gender category (0 or 1).
        ethnicity: Ethnicity category (0-5).
        device: Device to run on.
        max_length: Maximum sequence length (default: 50, aligned with training data).
        temperature: Sampling temperature (default: 0.8, slightly more conservative).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        min_codes_per_visit: Minimum diagnosis codes before allowing visit end.
        max_codes_per_visit: Maximum diagnosis codes per visit before forcing visit end.

    Returns:
        Generated sequence string with decoded diagnosis codes.
    """
    model.eval()

    # Prepare demographics
    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[gender, ethnicity]], dtype=torch.long).to(device)

    # Start with BOS token
    generated_ids = [tokenizer.bos_token_id]

    # Token IDs for special tokens
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")

    # Visit structure tracking
    inside_visit = False
    codes_in_current_visit = 0

    # Create dummy encoder input (will be replaced by prompts)
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    with torch.no_grad():
        # Generate tokens autoregressively
        for step in range(max_length):
            # Prepare decoder input
            decoder_input_ids = torch.tensor([generated_ids], dtype=torch.long).to(device)

            # Forward pass
            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                x_num=x_num,
                x_cat=x_cat
            )

            # Get next token logits
            next_token_logits = outputs.logits[0, -1, :] / temperature

            # Suppress immediate duplicate diagnosis codes
            if len(generated_ids) > 0:
                last_token_id = generated_ids[-1]
                if last_token_id >= tokenizer.code_offset:  # Only for diagnosis codes, not special tokens
                    next_token_logits[last_token_id] = float('-inf')

            # Always suppress BOS and EOS tokens during generation (should never be sampled)
            next_token_logits[tokenizer.bos_token_id] = float('-inf')
            next_token_logits[tokenizer.eos_token_id] = float('-inf')
            next_token_logits[tokenizer.pad_token_id] = float('-inf')

            # Apply visit structure constraints
            if not inside_visit:
                # Force <v> to start a visit (suppress all other tokens)
                mask = torch.ones_like(next_token_logits) * float('-inf')
                mask[v_token_id] = next_token_logits[v_token_id]
                next_token_logits = mask
            elif codes_in_current_visit >= max_codes_per_visit:
                # Force <\v> to end visit if too many codes
                mask = torch.ones_like(next_token_logits) * float('-inf')
                mask[v_end_token_id] = next_token_logits[v_end_token_id]
                next_token_logits = mask
            elif codes_in_current_visit < min_codes_per_visit:
                # Suppress <\v> and <END> until minimum codes reached
                next_token_logits[v_end_token_id] = float('-inf')
                next_token_logits[end_token_id] = float('-inf')
                next_token_logits[v_token_id] = float('-inf')  # No nested visits
            else:
                # Allow natural sampling but suppress nested <v>
                next_token_logits[v_token_id] = float('-inf')

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
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

            generated_ids.append(next_token_id)

            # Update visit structure state
            if next_token_id == v_token_id:
                inside_visit = True
                codes_in_current_visit = 0
            elif next_token_id == v_end_token_id:
                inside_visit = False
                codes_in_current_visit = 0
            elif next_token_id == end_token_id:
                # Stop generation at <END> token
                break
            elif inside_visit and next_token_id >= tokenizer.code_offset:
                # Count diagnosis codes
                codes_in_current_visit += 1

    # Decode sequence
    decoded_sequence = tokenizer.decode(generated_ids)

    return decoded_sequence


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

    # Define diverse demographics for generation
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
    ]

    for i, demo in enumerate(test_demographics, 1):
        logger.info(f"\n{'─' * 80}")

        # Decode demographics
        decoded_demo = decode_patient_demographics(
            demo["age"], demo["gender"], demo["ethnicity"]
        )

        logger.info(f"Patient {i}:")
        logger.info(f"  Demographics: {decoded_demo['age']}yo {decoded_demo['ethnicity']} {decoded_demo['gender']}")

        # Generate sequence with improved parameters
        sequence = generate_patient_sequence(
            model=model,
            tokenizer=tokenizer,
            age=demo["age"],
            gender=demo["gender"],
            ethnicity=demo["ethnicity"],
            device=device,
            max_length=50,  # Realistic based on training data
            temperature=0.8,  # Slightly more conservative
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
