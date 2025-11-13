"""
Test to verify loss calculation correctness.
Reproduces the exact loss calculation from the model to identify bugs.
"""
import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from hierarchical_dataset import HierarchicalEHRDataset, HierarchicalEHRDataCollator
from torch.utils.data import DataLoader, random_split
from prompt_bart_model import PromptBartWithDemographicPrediction
from transformers import BartConfig


def setup_logger():
    """Set up logging."""
    logger = logging.getLogger("test_loss")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def test_loss_calculation():
    """Test loss calculation on a small batch."""
    print("=" * 80)
    print("Loss Calculation Verification Test")
    print("=" * 80)

    # Load config
    config = Config.from_defaults()
    device = torch.device("cpu")  # Use CPU for debugging
    logger = setup_logger()

    # Load data (small subset)
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=1000
    )

    # Build hierarchy and tokenizer
    hierarchy = ICD9Hierarchy(vocab, logger=logger)
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)

    # Create dataset
    dataset = HierarchicalEHRDataset(patient_records, tokenizer, logger=logger)

    # Create collator
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

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collator
    )

    # Initialize model
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
    model.to(device)
    model.train()  # Important: set to training mode

    print(f"\nModel Configuration:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Category offset: {tokenizer.category_offset}")
    print(f"  Code offset: {tokenizer.code_offset}")
    print(f"  Age loss weight: {config.model.age_loss_weight}")
    print(f"  Sex loss weight: {config.model.sex_loss_weight}")

    # Get one batch
    batch = next(iter(loader))

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    x_num = batch['x_num'].to(device)
    x_cat = batch['x_cat'].to(device)

    print(f"\nBatch Shape:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  x_num (age): {x_num.shape}")
    print(f"  x_cat (sex): {x_cat.shape}")

    # Check sex distribution
    print(f"\nSex Distribution in Batch:")
    sex_values = x_cat[:, 0].cpu().numpy()
    unique, counts = torch.unique(x_cat[:, 0], return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Sex {val}: {count} patients ({100*count/len(sex_values):.1f}%)")

    # Check age distribution
    print(f"\nAge Distribution in Batch:")
    ages = x_num[:, 0].cpu().numpy()
    print(f"  Min age: {ages.min():.1f}")
    print(f"  Max age: {ages.max():.1f}")
    print(f"  Mean age: {ages.mean():.1f}")

    # Check label distribution
    print(f"\nLabel Token Analysis:")
    valid_labels = labels[labels >= 0]
    print(f"  Total tokens: {labels.numel()}")
    print(f"  Valid tokens (not -100): {valid_labels.numel()}")
    print(f"  Padding tokens (-100): {(labels == -100).sum().item()}")

    # Check code mask
    code_offset = tokenizer.category_offset  # Using category offset for hierarchical
    code_mask = (labels >= code_offset) & (labels != -100)
    print(f"  Code tokens (>= {code_offset}): {code_mask.sum().item()}")
    print(f"  Code mask ratio: {100 * code_mask.sum().item() / labels.numel():.2f}%")

    # Forward pass
    print(f"\n{'=' * 80}")
    print("Forward Pass")
    print("=" * 80)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat,
            code_offset=code_offset
        )

    print(f"\nLoss Values:")
    print(f"  Total loss: {outputs.loss.item():.6f}")

    if hasattr(outputs, 'lm_loss'):
        print(f"  LM loss: {outputs.lm_loss.item():.6f}")
        print(f"  Age loss: {outputs.age_loss.item():.6f}")
        print(f"  Sex loss: {outputs.sex_loss.item():.6f}")

        # Verify calculation
        calculated_total = (outputs.lm_loss.item() +
                          config.model.age_loss_weight * outputs.age_loss.item() +
                          config.model.sex_loss_weight * outputs.sex_loss.item())
        print(f"\nVerification:")
        print(f"  Calculated total: {calculated_total:.6f}")
        print(f"  Actual total: {outputs.loss.item():.6f}")
        print(f"  Match: {abs(calculated_total - outputs.loss.item()) < 1e-5}")

    # Now test with a trained checkpoint if available
    checkpoint_path = "/scratch/jalenj4/promptehr_checkpoints/best_hierarchical_model.pt"
    if Path(checkpoint_path).exists():
        print(f"\n{'=' * 80}")
        print("Testing with Trained Checkpoint")
        print("=" * 80)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                x_num=x_num,
                x_cat=x_cat,
                code_offset=code_offset
            )

        # Note: In eval mode, auxiliary losses won't be computed
        print(f"\nEval Mode - Only LM Loss:")
        print(f"  Loss: {outputs.loss.item():.6f}")

        # Switch back to train mode to get auxiliary losses
        model.train()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                x_num=x_num,
                x_cat=x_cat,
                code_offset=code_offset
            )

        print(f"\nTrain Mode - All Losses:")
        print(f"  Total loss: {outputs.loss.item():.6f}")
        if hasattr(outputs, 'lm_loss'):
            print(f"  LM loss: {outputs.lm_loss.item():.6f}")
            print(f"  Age loss: {outputs.age_loss.item():.6f}")
            print(f"  Sex loss: {outputs.sex_loss.item():.6f}")

            # Check if sex loss is truly zero
            if outputs.sex_loss.item() < 1e-10:
                print(f"\n⚠️  WARNING: Sex loss is essentially zero!")
                print(f"  This suggests the model has learned to perfectly predict sex")
                print(f"  OR there is a bug in the loss calculation")


if __name__ == "__main__":
    test_loss_calculation()
