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
from prompt_bart_model import PromptBartWithDemographicPrediction


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
) -> PromptBartWithDemographicPrediction:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        tokenizer: DiagnosisCodeTokenizer instance.
        config: Configuration object.
        device: Device to load model on.
        logger: Logger instance.

    Returns:
        Loaded PromptBartWithDemographicPrediction instance.
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Initialize model architecture
    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id

    model = PromptBartWithDemographicPrediction(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        d_hidden=config.model.d_hidden,
        prompt_length=config.model.prompt_length,
        age_loss_weight=config.model.age_loss_weight,
        sex_loss_weight=config.model.sex_loss_weight
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
    model: PromptBartWithDemographicPrediction,
    tokenizer: DiagnosisCodeTokenizer,
    target_patient: 'PatientRecord',
    device: torch.device,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
    prompt_prob: float = 0.0,
    max_codes_per_visit: int = 20
) -> dict:
    """Generate synthetic patient via conditional reconstruction (PromptEHR approach).

    Given a real patient from test set, randomly masks codes and reconstructs
    the full visit structure. Default prompt_prob=0.0 means zero-code-prompt
    generation (only demographics provided).

    Args:
        model: Trained PromptBartModel.
        tokenizer: DiagnosisCodeTokenizer instance.
        target_patient: PatientRecord from test set to reconstruct.
        device: Device to run on.
        temperature: Sampling temperature (default: 0.3).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        prompt_prob: Probability of keeping each code as prompt (default: 0.0 = zero prompts).
        max_codes_per_visit: Cap visit codes at this number (default: 20).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of generated code sequences
            - 'target_visits': List[List[str]] of original codes
            - 'prompt_codes': List[List[str]] of codes provided as prompts
            - 'demographics': dict of patient demographics
    """
    model.eval()

    # Extract demographics (race removed)
    age = target_patient.age
    gender = 1 if target_patient.gender == 'F' else 0

    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[gender]], dtype=torch.long).to(device)  # Gender only, race removed

    # Initialize accumulators
    generated_visits = []
    prompt_codes_per_visit = []

    # Create dummy encoder input (prompts are in decoder)
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")

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

            # Step 3: Encode prompt codes as decoder input
            prompt_token_ids = [tokenizer.bos_token_id, v_token_id]
            for code in prompt_codes:
                # Codes are in vocab, need to add code_offset to get token ID
                code_idx = tokenizer.vocab.code2idx[code]
                code_token_id = tokenizer.code_offset + code_idx
                prompt_token_ids.append(code_token_id)

            decoder_input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(device)

            # Step 4: Generate to reconstruct full visit using model.generate()
            max_new_tokens = num_codes + 2  # Target length

            # CRITICAL: Use model.generate() instead of manual loop
            # This automatically handles:
            # - Temperature/top-k/top-p sampling
            # - no_repeat_ngram_size to prevent duplicates
            # - Bad words filtering
            generated_ids = model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                x_num=x_num,
                x_cat=x_cat,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,  # CRITICAL: Disable beam search, use sampling only
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=1,  # CRITICAL: Prevents duplicate codes
                eos_token_id=v_end_token_id,  # Stop at </v>
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.bos_token_id]]  # Suppress BOS in generation
            )

            # Step 5: Extract generated codes
            # generated_ids shape: [batch=1, seq_len]
            visit_token_ids = generated_ids[0].cpu().tolist()

            # Extract code tokens (skip BOS, <v>, <\v>)
            generated_code_ids = [
                tid for tid in visit_token_ids
                if tid >= tokenizer.code_offset
            ]

            # Decode codes (convert token IDs back to diagnosis codes)
            generated_codes = []
            for tid in generated_code_ids:
                code_idx = tid - tokenizer.code_offset
                if code_idx < len(tokenizer.vocab):
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
            'gender': target_patient.gender
        }
    }


def sample_demographics(
    age_mean: float = 60.0,
    age_std: float = 20.0,
    male_prob: float = 0.56
) -> dict:
    """Sample realistic patient demographics.

    Samples demographics from distributions matching MIMIC-III ICU population.

    Args:
        age_mean: Mean age for normal distribution (default: 60).
        age_std: Standard deviation for age (default: 20).
        male_prob: Probability of male gender (default: 0.56).

    Returns:
        Dictionary with:
            - 'age': float in range [0, 90]
            - 'sex': int (0=Male, 1=Female)
            - 'sex_str': str ('M' or 'F')
    """
    # Sample age from normal distribution, clipped to [0, 90]
    age = np.random.normal(age_mean, age_std)
    age = np.clip(age, 0, 90)

    # Sample sex from binomial distribution
    sex = 0 if np.random.rand() < male_prob else 1
    sex_str = 'M' if sex == 0 else 'F'

    return {
        'age': float(age),
        'sex': sex,
        'sex_str': sex_str
    }


def generate_patient_from_demographics(
    model: PromptBartWithDemographicPrediction,
    tokenizer: DiagnosisCodeTokenizer,
    device: torch.device,
    age: Optional[float] = None,
    sex: Optional[int] = None,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    max_sequence_length: int = 256
) -> dict:
    """Generate patient sequence from demographics only (zero code prompts).

    This is demographic-conditional generation with no code prompts.
    Demographics (age, sex) are injected via prompt embeddings, but no
    diagnosis codes are provided. The model generates complete visit structure
    and all diagnosis codes from scratch.

    Note: This is NOT fully unconditional - demographics still condition the
    generation via prompt embeddings injected into encoder/decoder layers.

    Args:
        model: Trained PromptBartModel.
        tokenizer: DiagnosisCodeTokenizer instance.
        device: Device to run on.
        age: Patient age (if None, sampled from distribution).
        sex: Patient sex ID (0=M, 1=F; if None, sampled).
        temperature: Sampling temperature (default: 0.7).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        max_sequence_length: Maximum tokens to generate (default: 256).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of diagnosis codes
            - 'demographics': dict with 'age' and 'sex'
            - 'num_visits': int
            - 'num_codes': int
    """
    model.eval()

    # Sample demographics if not provided
    if age is None or sex is None:
        sampled_demo = sample_demographics()
        age = sampled_demo['age'] if age is None else age
        sex = sampled_demo['sex'] if sex is None else sex

    sex_str = 'M' if sex == 0 else 'F'

    # Prepare demographic tensors
    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[sex]], dtype=torch.long).to(device)

    # Create dummy encoder input (encoder not used for unconditional generation)
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    # Decoder starts with just BOS token (no prompt codes)
    decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

    # Special token IDs
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")

    with torch.no_grad():
        try:
            # Generate full patient sequence until <END> token
            generated_ids = model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                x_num=x_num,
                x_cat=x_cat,
                max_new_tokens=max_sequence_length,
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=1,  # Prevent duplicate codes within generation
                eos_token_id=end_token_id,  # Stop at <END> token
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.bos_token_id]]  # Suppress BOS in generation
            )

            # Extract token sequence
            token_ids = generated_ids[0].cpu().tolist()

            # Parse into visit structure
            visits = parse_sequence_to_visits(token_ids, tokenizer)

            # Calculate statistics
            num_visits = len(visits)
            num_codes = sum(len(visit) for visit in visits)

            return {
                'generated_visits': visits,
                'demographics': {
                    'age': age,
                    'sex': sex_str
                },
                'num_visits': num_visits,
                'num_codes': num_codes
            }

        except (RuntimeError, Exception) as e:
            # If generation fails, return empty result
            print(f"Generation failed: {e}")
            return {
                'generated_visits': [],
                'demographics': {
                    'age': age,
                    'sex': sex_str
                },
                'num_visits': 0,
                'num_codes': 0
            }


def parse_sequence_to_visits(
    token_ids: list[int],
    tokenizer: DiagnosisCodeTokenizer
) -> list[list[str]]:
    """Parse generated token sequence into visit structure.

    Extracts visits by splitting at <v> and <\v> markers, and decodes
    diagnosis codes within each visit.

    Args:
        token_ids: List of token IDs from model generation.
        tokenizer: DiagnosisCodeTokenizer instance.

    Returns:
        List of visits, where each visit is a list of ICD-9 code strings.

    Example:
        Input: [BOS, <v>, 401.9, 250.00, <\v>, <v>, 428.0, <\v>, <END>]
        Output: [['401.9', '250.00'], ['428.0']]
    """
    visits = []
    current_visit_codes = []

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")
    bos_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")

    in_visit = False

    for token_id in token_ids:
        if token_id == v_token_id:
            # Start of visit
            in_visit = True
            current_visit_codes = []
        elif token_id == v_end_token_id:
            # End of visit
            if in_visit:
                visits.append(current_visit_codes)
                in_visit = False
        elif token_id in [bos_token_id, end_token_id, tokenizer.pad_token_id]:
            # Skip special tokens
            continue
        elif in_visit and token_id >= tokenizer.code_offset:
            # Diagnosis code token
            code_idx = token_id - tokenizer.code_offset
            if code_idx < len(tokenizer.vocab):
                code = tokenizer.vocab.idx2code[code_idx]
                current_visit_codes.append(code)

    # Handle case where sequence ends without closing visit marker
    if in_visit and len(current_visit_codes) > 0:
        visits.append(current_visit_codes)

    return visits


def decode_patient_demographics(age: float, gender: int) -> dict[str, str]:
    """Decode demographics back to readable format.

    Args:
        age: Normalized age value.
        gender: Gender category index.

    Returns:
        Dictionary with decoded demographics.
    """
    # Gender mapping (from data_loader.py)
    gender_map = {0: "M", 1: "F"}  # Fixed: M=0, F=1

    return {
        "age": f"{age:.1f}",
        "gender": gender_map.get(gender, "UNKNOWN")
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

    # Use test set patients for conditional generation
    # Split patient_records into train/val/test
    num_total = len(patient_records)
    num_train = int(num_total * 0.7)
    num_val = int(num_total * 0.15)
    test_patients = patient_records[num_train + num_val:]

    # Generate reconstructions for first 20 test patients
    num_generate = min(20, len(test_patients))
    logger.info(f"\nGenerating reconstructions for {num_generate} test patients...")

    for i, target_patient in enumerate(test_patients[:num_generate], 1):
        logger.info(f"\n{'─' * 80}")

        # Decode demographics
        decoded_demo = decode_patient_demographics(
            target_patient.age, 1 if target_patient.gender == 'F' else 0
        )

        logger.info(f"Patient {i} (subject_id={target_patient.subject_id}):")
        logger.info(f"  Demographics: {decoded_demo['age']}yo {decoded_demo['gender']}")
        logger.info(f"  Target visits: {len(target_patient.visits)}")

        # Generate conditional reconstruction
        result = generate_patient_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            target_patient=target_patient,
            device=device,
            temperature=0.3,  # Low temperature for medical validity
            top_k=40,
            top_p=0.9,
            prompt_prob=0.5,  # Mask ~50% of codes
            max_codes_per_visit=20
        )

        # Display results
        for visit_idx, (generated, target, prompt) in enumerate(
            zip(result['generated_visits'], result['target_visits'], result['prompt_codes'])
        ):
            logger.info(f"\n  Visit {visit_idx + 1}:")
            logger.info(f"    Prompt codes ({len(prompt)}): {prompt[:5]}...")  # Show first 5
            logger.info(f"    Target codes ({len(target)}): {target[:5]}...")
            logger.info(f"    Generated codes ({len(generated)}): {generated[:5]}...")

            # Compute Jaccard similarity
            if len(target) > 0:
                intersection = len(set(generated) & set(target))
                union = len(set(generated) | set(target))
                jaccard = intersection / union if union > 0 else 0.0
                logger.info(f"    Jaccard similarity: {jaccard:.3f}")

    logger.info("\n" + "=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
