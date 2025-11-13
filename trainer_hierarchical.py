"""
Training pipeline for hierarchical PromptEHR model.
Uses category-based training with ICD-9 hierarchy.
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
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from hierarchical_dataset import HierarchicalEHRDataset, HierarchicalEHRDataCollator
from prompt_bart_model import PromptBartModel
from metrics import MetricsTracker, compute_perplexity, compute_temporal_perplexity
from cooccurrence_utils import build_cooccurrence_matrix, cooccurrence_loss_efficient


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging to file and console."""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger("trainer_hierarchical")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / "training_hierarchical.log")
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
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }

    # Save regular checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = Path(checkpoint_dir) / "best_hierarchical_model.pt"
        torch.save(checkpoint, best_path)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    config: Config,
    tokenizer: HierarchicalDiagnosisTokenizer,
    val_patient_records: list
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    metrics_tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            x_num = batch['x_num'].to(device)
            x_cat = batch['x_cat'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                x_num=x_num,
                x_cat=x_cat,
                code_offset=tokenizer.category_offset
            )

            update_dict = {'loss': outputs.loss.item()}

            if hasattr(outputs, 'lm_loss'):
                update_dict['lm_loss'] = outputs.lm_loss.item()
                update_dict['age_loss'] = outputs.age_loss.item()
                update_dict['sex_loss'] = outputs.sex_loss.item()

            metrics_tracker.update(**update_dict)

    val_metrics = metrics_tracker.get_average_metrics()

    val_summary = (f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Perplexity: {val_metrics['perplexity']:.4f}")

    if 'lm_loss' in val_metrics:
        val_summary += (f", LM: {val_metrics['lm_loss']:.4f}, "
                       f"Age: {val_metrics['age_loss']:.2f}, "
                       f"Sex: {val_metrics['sex_loss']:.3f}")

    logger.info(val_summary)

    return val_metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: Config,
    logger: logging.Logger,
    tokenizer: HierarchicalDiagnosisTokenizer,
    cooccur_matrix: Optional[torch.Tensor] = None
) -> dict[str, float]:
    """Train for one epoch."""
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
            x_cat=x_cat,
            code_offset=tokenizer.category_offset  # Use category offset for hierarchical training
        )

        # Get base loss (LM + age + sex)
        loss = outputs.loss

        # Add co-occurrence regularization loss if enabled
        cooccur_loss_value = 0.0
        if cooccur_matrix is not None and config.model.cooccurrence_loss_weight > 0:
            cooccur_loss = cooccurrence_loss_efficient(
                generated_code_ids=input_ids,
                cooccur_matrix=cooccur_matrix,
                tokenizer=tokenizer,
                threshold=5
            )
            loss = loss + config.model.cooccurrence_loss_weight * cooccur_loss
            cooccur_loss_value = cooccur_loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track metrics
        update_dict = {'loss': loss.item()}

        # Track auxiliary losses if available (multi-task learning)
        if hasattr(outputs, 'lm_loss'):
            update_dict['lm_loss'] = outputs.lm_loss.item()
            update_dict['age_loss'] = outputs.age_loss.item()
            update_dict['sex_loss'] = outputs.sex_loss.item()

        # Track co-occurrence loss if enabled
        if cooccur_loss_value > 0:
            update_dict['cooccur_loss'] = cooccur_loss_value

        metrics_tracker.update(**update_dict)

        # Update progress bar
        current_metrics = metrics_tracker.get_last_metrics()
        postfix = {
            'loss': f"{current_metrics['loss']:.4f}",
            'ppl': f"{current_metrics['perplexity']:.2f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        }

        # Add auxiliary losses to progress bar if available
        if 'lm_loss' in current_metrics:
            postfix['lm'] = f"{current_metrics['lm_loss']:.4f}"
            postfix['age'] = f"{current_metrics['age_loss']:.2f}"
            postfix['sex'] = f"{current_metrics['sex_loss']:.3f}"

        # Add co-occurrence loss to progress bar if available
        if 'cooccur_loss' in current_metrics:
            postfix['cooc'] = f"{current_metrics['cooccur_loss']:.3f}"

        progress_bar.set_postfix(postfix)

        # Log periodically
        if (step + 1) % config.training.log_every_n_steps == 0:
            avg_metrics = metrics_tracker.get_average_metrics()
            log_msg = (f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)} - "
                      f"Loss: {avg_metrics['loss']:.4f}, "
                      f"Perplexity: {avg_metrics['perplexity']:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Add auxiliary losses if available
            if 'lm_loss' in avg_metrics:
                log_msg += (f", LM: {avg_metrics['lm_loss']:.4f}, "
                           f"Age: {avg_metrics['age_loss']:.2f}, "
                           f"Sex: {avg_metrics['sex_loss']:.3f}")

            # Add co-occurrence loss if available
            if 'cooccur_loss' in avg_metrics:
                log_msg += f", Cooccur: {avg_metrics['cooccur_loss']:.3f}"

            logger.info(log_msg)

    # Get epoch average metrics
    epoch_metrics = metrics_tracker.get_average_metrics()

    epoch_summary = (f"Epoch {epoch+1} Complete - "
                    f"Avg Loss: {epoch_metrics['loss']:.4f}, "
                    f"Avg Perplexity: {epoch_metrics['perplexity']:.4f}")

    # Add auxiliary losses if available
    if 'lm_loss' in epoch_metrics:
        epoch_summary += (f", LM: {epoch_metrics['lm_loss']:.4f}, "
                         f"Age: {epoch_metrics['age_loss']:.2f}, "
                         f"Sex: {epoch_metrics['sex_loss']:.3f}")

    # Add co-occurrence loss if available
    if 'cooccur_loss' in epoch_metrics:
        epoch_summary += f", Cooccur: {epoch_metrics['cooccur_loss']:.3f}"

    logger.info(epoch_summary)

    return epoch_metrics


def main():
    """Main training function for hierarchical model."""
    # Load configuration
    config = Config.from_defaults()

    # Set up logging
    logger = setup_logging(config.training.log_dir)
    logger.info("=" * 80)
    logger.info("PromptEHR Hierarchical Training Pipeline")
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

    # Build ICD-9 hierarchy
    logger.info("\n" + "=" * 80)
    logger.info("Building ICD-9 Hierarchy")
    logger.info("=" * 80)

    hierarchy = ICD9Hierarchy(vocab, logger)

    # Create hierarchical tokenizer
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    logger.info(f"Hierarchical tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"  Category tokens: {tokenizer.get_n_categories()}")
    logger.info(f"  Code tokens: {tokenizer.get_n_codes()}")

    # Create dataset
    dataset = HierarchicalEHRDataset(patient_records, tokenizer, logger)
    logger.info(f"Dataset size: {len(dataset)} patients")

    # Train/validation split
    train_size = int((1 - config.data.train_val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")

    # Extract validation patient records for TPL computation
    val_patient_records = [patient_records[i] for i in val_dataset.indices]

    # Create data collator
    collator = HierarchicalEHRDataCollator(
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

    # Use multi-task learning model with auxiliary age/sex prediction
    from prompt_bart_model import PromptBartWithDemographicPrediction

    model = PromptBartWithDemographicPrediction(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        d_hidden=config.model.d_hidden,
        prompt_length=config.model.prompt_length,
        age_loss_weight=config.model.age_loss_weight,
        sex_loss_weight=config.model.sex_loss_weight
    )

    logger.info(f"Model: PromptBartWithDemographicPrediction (hierarchical training)")
    logger.info(f"  Age loss weight: {config.model.age_loss_weight}")
    logger.info(f"  Sex loss weight: {config.model.sex_loss_weight}")

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

    # Build co-occurrence matrix for regularization
    logger.info("\n" + "=" * 80)
    logger.info("Building Co-occurrence Matrix")
    logger.info("=" * 80)

    # Extract training patient records
    train_patient_records = [patient_records[i] for i in train_dataset.indices]

    # Build matrix from training data only
    cooccur_matrix = build_cooccurrence_matrix(train_patient_records, vocab, logger)
    logger.info(f"Co-occurrence weight: {config.model.cooccurrence_loss_weight}")

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
            logger=logger,
            tokenizer=tokenizer,
            cooccur_matrix=cooccur_matrix
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
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
