import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BartForConditionalGeneration, BartTokenizer, get_scheduler
from tqdm.auto import tqdm
import random
import logging
import sys
from datetime import datetime
import argparse

def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging to file and console with appropriate formatting.

    Args:
        log_file: Path to log file. If None, uses timestamp-based name.

    Returns:
        Configured logger instance.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"

    logger = logging.getLogger("ehr_generation")
    logger.setLevel(logging.DEBUG)

    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler with simpler formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --- 1. Configuration ---
ETHNICITIES = ['WHITE', 'BLACK', 'UNKNOWN/NOT SPECIFIED', 'HISPANIC OR LATINO', 'OTHER', 'ASIAN', 'UNABLE TO OBTAIN']

def get_default_config() -> dict:
    """Get default configuration for training.

    Returns:
        Configuration dictionary with hyperparameters.
    """
    return {
        "model_name": "facebook/bart-base",
        "num_patients": 3000,
        "max_seq_length": 256,
        "batch_size": 16,
        "num_epochs": 15,
        "learning_rate": 1e-4,
        "generation_temp": 1.0,
        "generation_max_length": 256,
        "num_warmup_steps": 0,
        "top_k": 50,
        "data_dir": "data_files",
        "mimic_paths": {
            "patients_path": "data_files/PATIENTS.csv",
            "admissions_path": "data_files/ADMISSIONS.csv",
            "diagnoses_path": "data_files/DIAGNOSES_ICD.csv"
        }
    }

# --- 2. Synthetic Data Generation ---
# In a real-world scenario, you would load your MIMIC dataset here.
# For this example, we generate synthetic data that follows the specified format.
def generate_synthetic_data(num_patients: int, logger: logging.Logger) -> list[str]:
    """Generates synthetic patient sequences for testing.

    Args:
        num_patients: Number of synthetic patient records to generate.
        logger: Logger instance for output.

    Returns:
        List of formatted patient sequence strings.
    """
    logger.info(f"Generating {num_patients} synthetic patient records for training")
    sequences = []

    # Predefined sample of mock ICD-10 codes
    mock_icd_codes = [f"{chr(random.randint(65, 90))}{random.randint(10, 99)}.{random.randint(0, 9)}" for _ in range(100)]

    for _ in range(num_patients):
        # a. Sample demographics
        age = np.random.randint(20, 91)
        sex = np.random.choice(['M', 'F'])
        race = np.random.choice(ETHNICITIES)

        # Start the sequence string
        # The <demo> tag is treated as a regular string part, not a special token
        sequence = f"{age} {race} {sex} <demo> "

        # b. Generate a random number of medical visits
        num_visits = np.random.randint(1, 6)

        for _ in range(num_visits):
            # c. For each visit, sample a random number of ICD codes
            num_codes = np.random.randint(1, 5)
            visit_codes = random.sample(mock_icd_codes, num_codes)

            # d. Format the visit string
            visit_str = " ".join(visit_codes)
            sequence += f"<v> {visit_str} <\\v> "

        # e. Add the end token
        sequence += "<END>"
        sequences.append(sequence)

    logger.info(f"Synthetic data generation complete: {len(sequences)} sequences")
    return sequences

import pandas as pd
import numpy as np

def load_and_format_mimic_data(
    patients_path: str,
    admissions_path: str,
    diagnoses_path: str,
    logger: logging.Logger,
    num_patients: int = None
) -> list[str]:
    """Loads MIMIC-III data and formats into patient sequences.

    Args:
        patients_path: Path to PATIENTS.csv file.
        admissions_path: Path to ADMISSIONS.csv file.
        diagnoses_path: Path to DIAGNOSES.csv file.
        logger: Logger instance for output.
        num_patients: Maximum number of patient sequences to generate.

    Returns:
        List of formatted patient sequence strings.
    """
    logger.info("Loading MIMIC-III data files")
    logger.debug(f"Patients: {patients_path}")
    logger.debug(f"Admissions: {admissions_path}")
    logger.debug(f"Diagnoses: {diagnoses_path}")

    try:
        # Load all three files
        patients_df = pd.read_csv(patients_path, parse_dates=['DOB'])
        logger.info(f"Loaded {len(patients_df)} patients")

        admissions_df = pd.read_csv(admissions_path, parse_dates=['ADMITTIME'])
        logger.info(f"Loaded {len(admissions_df)} admissions")

        diagnoses_df = pd.read_csv(diagnoses_path)
        logger.info(f"Loaded {len(diagnoses_df)} diagnosis records")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e.filename}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during file loading: {e}")
        return []

    # --- Step 1: Prepare Demographics (Patients and Admissions) ---

    # 1.1 Calculate Age at First Admission
    # Find the earliest admission time for each patient
    first_admissions = admissions_df.loc[
        admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
    ][['SUBJECT_ID', 'ADMITTIME']]

    # Merge DOB and First Admission Time
    demo_df = pd.merge(
        patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
        first_admissions,
        on='SUBJECT_ID',
        how='inner'
    )

    # Calculate Age: Difference in years between admission and DOB
    # MIMIC data often has censored ages > 89. We'll stick to simple year diff.
    demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
    # Cap age for privacy (as is common practice with MIMIC)
    demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])

    # 1.2 Select relevant columns from Admissions (Ethnicity)
    admissions_info = admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY']]

    # --- Step 2: Merge All Data ---

    # 2.1 Merge Admissions and Diagnoses
    merged_visits_df = pd.merge(
        admissions_info,
        diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    # 2.2 Final Merge with Demographics
    # Merge visit data with the calculated age and gender
    final_df = pd.merge(
        merged_visits_df,
        demo_df[['SUBJECT_ID', 'AGE', 'GENDER']],
        on='SUBJECT_ID',
        how='left' # Use left to keep all visit data
    )

    # --- Step 3: Format Sequences ---

    # Sort chronologically by patient, admission time, and ICD code sequence number
    final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)

    logger.info("Data processing complete. Formatting sequences...")

    sequences = []
    # Group by patient
    patient_groups = final_df.groupby('SUBJECT_ID')

    for _, patient_data in patient_groups:

        # a. Extract demographics from the first row of patient data
        # Age and Gender are consistent for a patient across all visits in this setup
        age = str(patient_data['AGE'].iloc[0]) if not pd.isna(patient_data['AGE'].iloc[0]) else 'Unknown'
        sex = patient_data['GENDER'].iloc[0] if not pd.isna(patient_data['GENDER'].iloc[0]) else 'Unknown'

        # Race/Ethnicity: Use the most frequent ethnicity recorded for the patient
        race = patient_data['ETHNICITY'].mode().iloc[0] if not patient_data['ETHNICITY'].mode().empty else 'Unknown'

        # Start the sequence string
        sequence = f"{age} {race} {sex} <demo> "

        # b. Group by visit (HADM_ID) for this patient
        # The chronological sort ensures the visits are processed in order
        visit_groups = patient_data.groupby('HADM_ID', sort=False)

        for _, visit_data in visit_groups:
            # c. Get ICD codes for the visit, maintaining the order defined by SEQ_NUM
            icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()

            # d. Format the visit string
            visit_str = " ".join(icd_codes)
            sequence += f"<v> {visit_str} <\\v> "

        # e. Add the end token
        sequence += "<END>"
        sequences.append(sequence)

        # Truncate if num_patients limit is reached
        if num_patients is not None and len(sequences) >= num_patients:
            break

    logger.info(f"Sequence formatting complete. Generated {len(sequences)} patient sequences")

    # Log sample sequences for debugging
    if len(sequences) > 0:
        logger.debug(f"Sample sequence: {sequences[0][:200]}...")

        # Analyze sequence statistics
        avg_visits = sum(seq.count('<v>') for seq in sequences) / len(sequences)
        avg_codes_per_seq = sum(len(seq.split()) for seq in sequences) / len(sequences)
        logger.info(f"Average visits per patient: {avg_visits:.2f}")
        logger.info(f"Average tokens per sequence: {avg_codes_per_seq:.2f}")

    return sequences

# --- 3. PyTorch Dataset ---
class PatientSequenceDataset(Dataset):
    """Custom PyTorch Dataset to handle tokenization of patient sequences."""

    def __init__(self, sequences: list[str], tokenizer: BartTokenizer, max_length: int, logger: logging.Logger):
        """Initialize dataset with sequences and tokenizer.

        Args:
            sequences: List of patient sequence strings.
            tokenizer: BART tokenizer instance.
            max_length: Maximum sequence length for padding/truncation.
            logger: Logger instance for output.
        """
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.logger = logger

        # Log tokenization statistics for first sequence
        if len(sequences) > 0:
            sample_tokens = tokenizer(sequences[0], return_tensors="pt")
            self.logger.debug(f"First sequence length: {sample_tokens.input_ids.shape[1]} tokens")
            self.logger.debug(f"Pad token ID: {self.pad_token_id}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.sequences[idx]

        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        # --- CRITICAL: Mask padding tokens in the labels ---
        # Set all pad tokens in the labels to -100 so they're ignored in loss calculation
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        # -----------------------------------------------------------

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --- 4. Model Training ---
def train_model(
    model: BartForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: AdamW,
    lr_scheduler,
    device: torch.device,
    num_epochs: int,
    logger: logging.Logger
) -> None:
    """Main training loop for the BART model.

    Args:
        model: BART model instance.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        lr_scheduler: Learning rate scheduler.
        device: Device to train on (CPU/GPU).
        num_epochs: Number of training epochs.
        logger: Logger instance for output.
    """
    model.to(device)
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    logger.info(f"Starting model training for {num_epochs} epochs")
    logger.info(f"Training steps: {num_training_steps}")
    logger.info(f"Device: {device}")

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to the appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Log detailed diagnostics for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                logger.debug(f"First batch input_ids shape: {batch['input_ids'].shape}")
                logger.debug(f"First batch labels shape: {batch['labels'].shape}")
                logger.debug(f"First batch attention_mask shape: {batch['attention_mask'].shape}")
                logger.debug(f"Initial loss: {loss.item():.4f}")

                # Check for special tokens in first batch
                special_token_counts = {
                    'pad': (batch['input_ids'] == 1).sum().item(),
                    'masked_labels': (batch['labels'] == -100).sum().item()
                }
                logger.debug(f"Special tokens in first batch: {special_token_counts}")

            # Backward pass
            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            batch_count += 1

            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

        # Log epoch statistics
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed | Avg Loss: {avg_epoch_loss:.4f}")

    logger.info("Training complete")
    logger.info(f"Final average loss: {epoch_losses[-1]:.4f}")
    logger.debug(f"Loss history: {epoch_losses}")


# --- 5. Generation/Sampling Function ---
def generate_patient_sequence(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    age: int,
    race: str,
    sex: str,
    device: torch.device,
    temp: float,
    max_len: int,
    top_k: int,
    logger: logging.Logger
) -> str:
    """Generates a new patient sequence starting from demographic data.

    Args:
        model: Trained BART model.
        tokenizer: BART tokenizer.
        age: Patient age.
        race: Patient race/ethnicity.
        sex: Patient sex (M/F).
        device: Device to run on.
        temp: Sampling temperature.
        max_len: Maximum generation length.
        top_k: Top-k sampling parameter.
        logger: Logger instance.

    Returns:
        Generated patient sequence string.
    """
    model.eval()
    model.to(device)

    # Create the initial prompt
    prompt = f"{age} {race} {sex} <demo>"
    logger.info(f"Generating sequence for prompt: '{prompt}'")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logger.debug(f"Prompt token IDs: {inputs.input_ids[0].tolist()}")
    logger.debug(f"Prompt tokens: {[tokenizer.decode([t]) for t in inputs.input_ids[0]]}")

    # Get special token IDs
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    logger.debug(f"Special token IDs - <v>: {v_token_id}, <\\v>: {v_end_token_id}, <END>: {end_token_id}, PAD: {pad_token_id}, EOS: {eos_token_id}")

    # Initialize decoder input to the prompt tokens + the forced <v> token
    initial_decoder_input_ids = torch.cat([
        inputs.input_ids,
        torch.tensor([[v_token_id]], device=device)
    ], dim=1)
    logger.debug(f"Initial decoder input (with forced <v>): {initial_decoder_input_ids[0].tolist()}")

    # --- DIAGNOSTIC: Analyze model's next token predictions ---
    with torch.no_grad():
        # Get raw logits for next token after prompt
        outputs = model(
            input_ids=inputs.input_ids,
            decoder_input_ids=inputs.input_ids
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze()

        # Log probabilities for critical tokens
        logger.debug(f"After prompt, probability of <v>: {next_token_probs[v_token_id].item():.6f}")
        logger.debug(f"After prompt, probability of <END>: {next_token_probs[end_token_id].item():.6f}")
        logger.debug(f"After prompt, probability of PAD: {next_token_probs[pad_token_id].item():.6f}")
        logger.debug(f"After prompt, probability of EOS: {next_token_probs[eos_token_id].item():.6f}")

        # Find top 20 predicted tokens
        top_20_probs = next_token_probs.topk(20)
        logger.debug("Top 20 next token predictions after prompt:")
        for i in range(20):
            token_id = top_20_probs.indices[i].item()
            token_prob = top_20_probs.values[i].item()
            token_str = tokenizer.decode(token_id)

            # Highlight important tokens
            if token_id == end_token_id:
                token_str = f"<END>"
            elif token_id == v_token_id:
                token_str = f"<v>"
            elif token_id == v_end_token_id:
                token_str = f"<\\v>"
            elif token_id == pad_token_id:
                token_str = f"<PAD>"
            elif token_id == eos_token_id:
                token_str = f"<EOS>"

            logger.debug(f"  Rank {i+1}: {token_str:<20} | Prob: {token_prob:.6f} | ID: {token_id}")

    # --- GENERATION ---
    with torch.no_grad():
        logger.info(f"Starting generation with temp={temp}, top_k={top_k}, max_length={max_len}")

        output_ids = model.generate(
            inputs.input_ids,
            decoder_input_ids=initial_decoder_input_ids,
            max_length=max_len + 1,
            do_sample=True,
            temperature=temp,
            top_k=top_k,
            eos_token_id=end_token_id,
            pad_token_id=pad_token_id,
            bad_words_ids=[[eos_token_id]]  # Block BART's EOS
        )

        logger.debug(f"Generated token IDs: {output_ids[0].tolist()}")
        logger.debug(f"Generated length: {output_ids.shape[1]} tokens")

        # Analyze generated tokens
        generated_tokens = output_ids[0].tolist()
        v_count = generated_tokens.count(v_token_id)
        v_end_count = generated_tokens.count(v_end_token_id)
        end_count = generated_tokens.count(end_token_id)

        logger.info(f"Generated token counts - <v>: {v_count}, <\\v>: {v_end_count}, <END>: {end_count}")

        # Check if generation only contains prompt + forced token
        if len(generated_tokens) <= len(initial_decoder_input_ids[0]):
            logger.warning("Generation did not extend beyond forced initial tokens!")

    # Decode the generated token IDs
    generated_sequence = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    logger.debug(f"Raw generated sequence: {generated_sequence}")

    generated_sequence_clean = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Generated sequence (clean): {generated_sequence_clean}")

    return generated_sequence_clean


# --- Main Execution Block ---
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train BART model for synthetic EHR generation")
    parser.add_argument("--num_patients", type=int, default=None, help="Number of patients to use for training")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--generation_temp", type=float, default=None, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")
    parser.add_argument("--num_warmup_steps", type=int, default=None, help="Number of warmup steps for LR scheduler")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("=== EHR Generation Training Started ===")

    # Load and update configuration
    config = get_default_config()
    if args.num_patients is not None:
        config['num_patients'] = args.num_patients
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.generation_temp is not None:
        config['generation_temp'] = args.generation_temp
    if args.top_k is not None:
        config['top_k'] = args.top_k
    if args.max_seq_length is not None:
        config['max_seq_length'] = args.max_seq_length
    if args.num_warmup_steps is not None:
        config['num_warmup_steps'] = args.num_warmup_steps

    logger.info(f"Configuration: {config}")

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Step 1: Load Tokenizer and Model, and add special tokens ---
    logger.info(f"Loading pretrained BART model: {config['model_name']}")
    tokenizer = BartTokenizer.from_pretrained(config['model_name'])
    model = BartForConditionalGeneration.from_pretrained(config['model_name'])

    # Add special tokens - <demo> prevents fragmentation
    special_tokens = ["<demo>", "<v>", "<\\v>", "<END>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    logger.info(f"Added {num_added} special tokens: <demo>, <v>, <\\v>, <END>")

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # Initialize new token embeddings from pretrained mean
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        mean_embed = embed_weight[:-num_added].mean(dim=0)
        for i in range(num_added):
            embed_weight[-(num_added - i)] = mean_embed
    logger.info(f"Initialized {num_added} new embeddings from pretrained mean")

    logger.debug(f"Token ID for <demo>: {tokenizer.convert_tokens_to_ids('<demo>')}")
    logger.debug(f"Token ID for <v>: {tokenizer.convert_tokens_to_ids('<v>')}")
    logger.debug(f"Token ID for <\\v>: {tokenizer.convert_tokens_to_ids('<\\v>')}")
    logger.debug(f"Token ID for <END>: {tokenizer.convert_tokens_to_ids('<END>')}")

    # --- Step 2: Prepare Data ---
    sequences = load_and_format_mimic_data(
        **config["mimic_paths"],
        logger=logger,
        num_patients=config['num_patients']
    )

    if len(sequences) == 0:
        logger.error("No sequences loaded. Exiting.")
        sys.exit(1)

    # Log sample sequences
    logger.info("Sample patient sequences:")
    for i, seq in enumerate(random.sample(sequences, min(5, len(sequences)))):
        logger.info(f"Sample {i+1}: {seq[:150]}...")

    dataset = PatientSequenceDataset(sequences, tokenizer, config['max_seq_length'], logger)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    logger.info(f"Created dataset with {len(dataset)} samples, {len(dataloader)} batches")

    # --- Step 3: Set up Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    num_training_steps = config['num_epochs'] * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=num_training_steps
    )
    logger.info(f"Optimizer: AdamW with lr={config['learning_rate']}")
    logger.info(f"Scheduler: linear with {config['num_warmup_steps']} warmup steps")

    # --- Step 4: Train the model ---
    train_model(
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        device,
        config['num_epochs'],
        logger
    )

    # --- Step 5: Sample new data ---
    logger.info("=== Generating synthetic patient data ===")

    # Generate 5 samples with detailed diagnostics
    num_samples = 5
    for i in range(num_samples):
        sample_age = np.random.randint(20, 91)
        sample_sex = np.random.choice(['M', 'F'])
        sample_race = np.random.choice(ETHNICITIES)

        logger.info(f"\n--- Sample {i+1}/{num_samples} ---")
        generated_text = generate_patient_sequence(
            model=model,
            tokenizer=tokenizer,
            age=sample_age,
            race=sample_race,
            sex=sample_sex,
            device=device,
            temp=config['generation_temp'],
            max_len=config['generation_max_length'],
            top_k=config['top_k'],
            logger=logger
        )

    logger.info("=== Training and generation complete ===")