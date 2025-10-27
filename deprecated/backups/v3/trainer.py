"""
Training pipeline for PromptEHR model.
Integrates Phase 1 data with Phase 2 model architecture.
"""
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import BartConfig, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from dataset import EHRPatientDataset, EHRDataCollator
from prompt_bart_model import PromptBartModel
from metrics import MetricsTracker, compute_perplexity, compute_temporal_perplexity


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging to file and console.

    Args:
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / "training.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
    keep_last_n: int = 2
):
    """Save model checkpoint and clean up old ones.

    Args:
        model: PromptBartModel instance.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        val_loss: Validation loss.
        checkpoint_dir: Directory to save checkpoints.
        is_best: Whether this is the best model so far.
        keep_last_n: Number of recent checkpoints to keep (default: 2).
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir_path / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir_path / "best_model.pt"
        torch.save(checkpoint, best_path)

    # Clean up old checkpoints (keep only last N + best_model.pt)
    epoch_checkpoints = sorted(
        checkpoint_dir_path.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    # Keep only the last N checkpoints
    if len(epoch_checkpoints) > keep_last_n:
        for old_checkpoint in epoch_checkpoints[:-keep_last_n]:
            old_checkpoint.unlink()  # Delete old checkpoint


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
) -> int:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: PromptBartModel instance.
        optimizer: Optimizer (optional).
        scheduler: Learning rate scheduler (optional).

    Returns:
        Epoch number from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    config: Config,
    tokenizer: DiagnosisCodeTokenizer,
    val_patient_records: list
) -> dict[str, float]:
    """Run validation loop with TPL computation.

    Args:
        model: PromptBartModel instance.
        val_loader: Validation DataLoader.
        device: Device to run on.
        logger: Logger instance.
        config: Configuration object.
        tokenizer: DiagnosisCodeTokenizer instance.
        val_patient_records: List of validation patient records for TPL.

    Returns:
        Dictionary with validation metrics including TPL.
    """
    model.eval()
    metrics_tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            x_num = batch['x_num'].to(device)
            x_cat = batch['x_cat'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                x_num=x_num,
                x_cat=x_cat
            )

            # Track metrics
            metrics_tracker.update(
                loss=outputs.loss.item(),
                logits=outputs.logits,
                labels=labels,
                compute_accuracy=True
            )

    # Get average metrics
    val_metrics = metrics_tracker.get_average_metrics()

    logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                f"Perplexity: {val_metrics['perplexity']:.4f}, "
                f"Token Accuracy: {val_metrics.get('token_accuracy', 0):.4f}, "
                f"Code Accuracy: {val_metrics.get('code_accuracy', 0):.4f}")

    # Compute TPL if enabled
    if config.training.compute_tpl:
        tpl = compute_temporal_perplexity(
            model=model,
            patient_records=val_patient_records,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            max_samples=500
        )
        val_metrics['tpl'] = tpl
        logger.info(f"Temporal Perplexity (TPL): {tpl:.4f}")

    return val_metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: Config,
    logger: logging.Logger
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: PromptBartModel instance.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run on.
        epoch: Current epoch number.
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Dictionary with training metrics.
    """
    model.train()
    metrics_tracker = MetricsTracker()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        x_num = batch['x_num'].to(device)
        x_cat = batch['x_cat'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track metrics
        metrics_tracker.update(loss=loss.item())

        # Update progress bar
        current_metrics = metrics_tracker.get_last_metrics()
        progress_bar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'ppl': f"{current_metrics['perplexity']:.2f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # Log periodically
        if (step + 1) % config.training.log_every_n_steps == 0:
            avg_metrics = metrics_tracker.get_average_metrics()
            logger.info(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)} - "
                       f"Loss: {avg_metrics['loss']:.4f}, "
                       f"Perplexity: {avg_metrics['perplexity']:.4f}, "
                       f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # Get epoch average metrics
    epoch_metrics = metrics_tracker.get_average_metrics()

    logger.info(f"Epoch {epoch+1} Complete - "
                f"Avg Loss: {epoch_metrics['loss']:.4f}, "
                f"Avg Perplexity: {epoch_metrics['perplexity']:.4f}")

    return epoch_metrics


def main():
    """Main training function."""
    # Load configuration
    config = Config.from_defaults()

    # Set up logging
    logger = setup_logging(config.training.log_dir)
    logger.info("=" * 80)
    logger.info("PromptEHR Training Pipeline - Phase 3")
    logger.info("=" * 80)
    logger.info(f"\n{config}")

    # Set random seed
    set_seed(config.training.seed)
    logger.info(f"\nRandom seed: {config.training.seed}")

    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load MIMIC-III data
    logger.info("\n" + "=" * 80)
    logger.info("Loading MIMIC-III Data")
    logger.info("=" * 80)

    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    logger.info(f"Loaded {len(patient_records)} patients")
    logger.info(f"Vocabulary size: {len(vocab)} diagnosis codes")

    # Create tokenizer
    tokenizer = DiagnosisCodeTokenizer(vocab)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create dataset
    dataset = EHRPatientDataset(patient_records, tokenizer, logger)
    logger.info(f"Dataset size: {len(dataset)} patients")

    # Train/validation split
    train_size = int((1 - config.data.train_val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")

    # Extract validation patient records for TPL computation
    val_patient_records = [patient_records[i] for i in val_dataset.indices]

    # Create data collator with corruption parameters
    collator = EHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=logger,
        lambda_poisson=config.training.lambda_poisson,
        del_probability=config.training.del_probability,
        rep_probability=config.training.rep_probability,
        corruption_prob=config.training.corruption_prob,
        use_mask_infilling=config.training.use_mask_infilling,
        use_token_deletion=config.training.use_token_deletion,
        use_token_replacement=config.training.use_token_replacement,
        use_next_visit_prediction=config.training.use_next_visit_prediction
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator
    )

    logger.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Initialize model
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Model")
    logger.info("=" * 80)

    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id

    logger.info(f"BART config vocab size: {bart_config.vocab_size}")

    model = PromptBartModel(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        d_hidden=config.model.d_hidden,
        prompt_length=config.model.prompt_length
    )

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Set up scheduler
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {config.training.warmup_steps}")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(config.training.num_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            config=config,
            logger=logger
        )

        # Validate
        if (epoch + 1) % config.training.validate_every_n_epochs == 0:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                logger=logger,
                config=config,
                tokenizer=tokenizer,
                val_patient_records=val_patient_records
            )

            # Check if best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                logger.info(f"New best model! Validation loss: {best_val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % config.training.save_every_n_epochs == 0 or is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    val_loss=val_metrics['loss'],
                    checkpoint_dir=config.training.checkpoint_dir,
                    is_best=is_best
                )
                logger.info(f"Checkpoint saved: epoch {epoch + 1}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation perplexity: {compute_perplexity(best_val_loss):.4f}")


if __name__ == "__main__":
    main()
