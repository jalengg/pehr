"""
Diagnose why sex loss went to zero.
Tests whether the model learned a trivial solution for sex prediction.
"""
import torch
import logging
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from hierarchical_dataset import HierarchicalEHRDataset, HierarchicalEHRDataCollator
from torch.utils.data import DataLoader
from prompt_bart_model import PromptBartWithDemographicPrediction
from transformers import BartConfig


# Create dummy logger
logging.basicConfig(level=logging.WARNING)
dummy_logger = logging.getLogger("dummy")


def diagnose_sex_prediction():
    """Diagnose sex prediction behavior."""
    print("=" * 80)
    print("Sex Prediction Diagnostic")
    print("=" * 80)

    # Load config
    config = Config.from_defaults()
    device = torch.device("cpu")

    # Load FULL dataset to match checkpoint vocab
    print("\nLoading full dataset...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=dummy_logger,
        num_patients=50000  # Match training
    )
    print(f"Loaded {len(patient_records)} patients, {len(vocab)} codes")

    # Build hierarchy and tokenizer
    hierarchy = ICD9Hierarchy(vocab, logger=dummy_logger)
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    print(f"Vocab size: {len(tokenizer)}")

    # Create dataset
    dataset = HierarchicalEHRDataset(patient_records, tokenizer, logger=dummy_logger)

    # Create collator
    collator = HierarchicalEHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=dummy_logger,
        lambda_poisson=0.0,  # No corruption
        corruption_prob=0.0,  # No corruption
        use_mask_infilling=False,
        use_token_deletion=False,
        use_token_replacement=False,
        use_next_visit_prediction=True
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
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

    # Load trained checkpoint
    checkpoint_path = "/scratch/jalenj4/promptehr_checkpoints/best_hierarchical_model.pt"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()  # Enable auxiliary loss computation

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Analyze sex prediction on first few batches
    print("\n" + "=" * 80)
    print("Sex Prediction Analysis")
    print("=" * 80)

    total_samples = 0
    total_correct = 0
    total_sex_loss = 0.0
    sex_0_count = 0
    sex_1_count = 0
    sex_0_correct = 0
    sex_1_correct = 0

    max_batches = 50  # Test on first 50 batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

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
                code_offset=tokenizer.category_offset
            )

            # Get sex predictions
            decoder_hiddens = outputs.decoder_hidden_states[-1]
            pooled_repr = decoder_hiddens.mean(dim=1)
            predicted_sex_logits = model.sex_predictor(pooled_repr)
            predicted_sex = predicted_sex_logits.argmax(dim=1)
            true_sex = x_cat[:, 0]

            # Compute accuracy
            correct = (predicted_sex == true_sex).sum().item()
            total_correct += correct
            total_samples += len(true_sex)

            # Track by sex
            for pred, true in zip(predicted_sex, true_sex):
                if true == 0:
                    sex_0_count += 1
                    if pred == true:
                        sex_0_correct += 1
                else:
                    sex_1_count += 1
                    if pred == true:
                        sex_1_correct += 1

            # Accumulate sex loss
            if hasattr(outputs, 'sex_loss'):
                total_sex_loss += outputs.sex_loss.item()

    # Print results
    accuracy = 100 * total_correct / total_samples
    print(f"\nResults on {total_samples} samples ({max_batches} batches):")
    print(f"  Overall sex prediction accuracy: {accuracy:.2f}%")
    print(f"  Sex 0 (M) accuracy: {100 * sex_0_correct / sex_0_count:.2f}% ({sex_0_correct}/{sex_0_count})")
    print(f"  Sex 1 (F) accuracy: {100 * sex_1_correct / sex_1_count:.2f}% ({sex_1_correct}/{sex_1_count})")
    print(f"  Average sex loss: {total_sex_loss / max_batches:.6f}")

    # Check for trivial solution
    print(f"\n" + "=" * 80)
    print("Diagnosis:")
    print("=" * 80)

    if accuracy > 99.5:
        print("✓ Sex prediction is nearly perfect (>99.5% accuracy)")
        print("  This explains why sex loss → 0.000")
        print()
        print("  Possible reasons:")
        print("  1. Sex-specific codes in dataset (e.g., pregnancy, prostate)")
        print("  2. Model learned strong sex-code associations")
        print("  3. Dataset has sex-biased code distributions")
        print()
        print("  This is NOT a bug, but indicates:")
        print("  - Sex prediction task is too easy")
        print("  - Sex loss provides no useful gradient after epoch 14")
        print("  - Consider removing sex loss or using different weight schedule")
    elif accuracy > 90:
        print("⚠ Sex prediction is very good (>90% accuracy)")
        print(f"  But sex loss is {total_sex_loss / max_batches:.6f}, not zero")
        print("  This suggests sex loss is still contributing")
    else:
        print(f"⚠ Sex prediction accuracy is only {accuracy:.2f}%")
        print("  But training logs show sex loss = 0.000")
        print("  This indicates a BUG in loss calculation or logging")

    # Check dataset sex distribution
    print(f"\n" + "=" * 80)
    print("Dataset Sex Distribution:")
    print("=" * 80)

    sex_distribution = {"M": 0, "F": 0}
    for record in patient_records[:1000]:
        sex_distribution[record.gender] += 1

    total = sum(sex_distribution.values())
    print(f"  Male (M): {sex_distribution['M']} ({100 * sex_distribution['M'] / total:.1f}%)")
    print(f"  Female (F): {sex_distribution['F']} ({100 * sex_distribution['F'] / total:.1f}%)")


if __name__ == "__main__":
    diagnose_sex_prediction()
