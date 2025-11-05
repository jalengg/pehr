"""
Test script to verify co-occurrence loss integration.
Runs a minimal training loop to ensure everything works.
"""
import torch
import logging
from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from dataset import EHRPatientDataset, EHRDataCollator
from torch.utils.data import DataLoader
from transformers import BartConfig
from prompt_bart_model import PromptBartWithDemographicPrediction
from cooccurrence_utils import build_cooccurrence_matrix, cooccurrence_loss_efficient


def test_cooccurrence_integration():
    """Test that co-occurrence loss integration works."""
    print("=" * 80)
    print("Testing Co-occurrence Loss Integration")
    print("=" * 80)

    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    # Use small dataset for testing
    config = Config.from_defaults()
    config.data.num_patients = 1000  # Small sample

    # Load data
    print("\n1. Loading data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )
    print(f"   Loaded {len(patient_records)} patients, {len(vocab)} codes")

    # Create tokenizer
    print("\n2. Creating tokenizer...")
    tokenizer = DiagnosisCodeTokenizer(vocab)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")

    # Build co-occurrence matrix
    print("\n3. Building co-occurrence matrix...")
    cooccur_matrix = build_cooccurrence_matrix(patient_records, vocab, logger=logger)
    print(f"   Matrix shape: {cooccur_matrix.shape}")
    print(f"   Non-zero entries: {(cooccur_matrix > 0).sum().item():,}")

    # Create dataset
    print("\n4. Creating dataset...")
    dataset = EHRPatientDataset(patient_records, tokenizer, logger=logger)
    collator = EHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=None,
        lambda_poisson=config.training.lambda_poisson,
        del_probability=config.training.del_probability,
        rep_probability=config.training.rep_probability,
        corruption_prob=config.training.corruption_prob,
        use_mask_infilling=config.training.use_mask_infilling,
        use_token_deletion=config.training.use_token_deletion,
        use_token_replacement=config.training.use_token_replacement,
        use_next_visit_prediction=config.training.use_next_visit_prediction
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)
    print(f"   Dataset size: {len(dataset)}")

    # Initialize model
    print("\n5. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    model.train()
    print(f"   Model on device: {device}")

    # Test one training step
    print("\n6. Testing training step with co-occurrence loss...")
    batch = next(iter(loader))
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
        code_offset=tokenizer.code_offset
    )

    base_loss = outputs.loss.item()
    print(f"   Base loss (LM + age + sex): {base_loss:.4f}")

    # Compute co-occurrence loss
    cooccur_matrix_device = cooccur_matrix.to(device)
    cooccur_loss = cooccurrence_loss_efficient(
        generated_code_ids=input_ids,
        cooccur_matrix=cooccur_matrix_device,
        tokenizer=tokenizer,
        threshold=5
    )

    cooccur_loss_value = cooccur_loss.item()
    print(f"   Co-occurrence loss: {cooccur_loss_value:.4f}")

    # Combined loss
    total_loss = outputs.loss + config.model.cooccurrence_loss_weight * cooccur_loss
    total_loss_value = total_loss.item()
    print(f"   Total loss (with cooccur weight {config.model.cooccurrence_loss_weight}): {total_loss_value:.4f}")

    # Verify loss is reasonable
    assert cooccur_loss_value >= 0, "Co-occurrence loss should be non-negative"
    assert total_loss_value > base_loss, "Total loss should be higher than base loss"

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nCo-occurrence loss integration is working correctly.")
    print(f"Ready to train with {config.data.num_patients} patients.")


if __name__ == "__main__":
    test_cooccurrence_integration()
