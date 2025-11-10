"""
Test hierarchical model integration.

Verifies that the existing PromptBartWithDemographicPrediction model
can work with hierarchical category-based training without modification.
"""
import torch
import logging
from transformers import BartConfig
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from hierarchical_dataset import HierarchicalEHRDataset, HierarchicalEHRDataCollator
from torch.utils.data import DataLoader
from prompt_bart_model import PromptBartWithDemographicPrediction


def test_hierarchical_model():
    """Test model with hierarchical dataset."""
    print("=" * 80)
    print("Testing Hierarchical Model Integration")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    # Load data
    config = Config.from_defaults()
    config.data.num_patients = 100

    print("\n1. Loading data and building hierarchy...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    hierarchy = ICD9Hierarchy(vocab, logger=None)
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    print(f"   Categories: {tokenizer.get_n_categories()}")

    # Create dataset and dataloader
    print("\n2. Creating dataset and dataloader...")
    dataset = HierarchicalEHRDataset(patient_records, tokenizer, logger)
    collator = HierarchicalEHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=logger
    )
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    print(f"   Dataset size: {len(dataset)}")

    # Initialize model
    print("\n3. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)  # Use hierarchical tokenizer vocab size
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
    print(f"   Model initialized on {device}")
    print(f"   Model vocab size: {bart_config.vocab_size}")

    # Test forward pass
    print("\n4. Testing forward pass...")
    batch = next(iter(dataloader))

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    x_num = batch['x_num'].to(device)
    x_cat = batch['x_cat'].to(device)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        x_num=x_num,
        x_cat=x_cat,
        code_offset=tokenizer.category_offset  # Use category_offset for masking
    )

    print(f"   Total loss: {outputs.loss.item():.4f}")
    print(f"   LM loss: {outputs.lm_loss.item():.4f}")
    print(f"   Age loss: {outputs.age_loss.item():.4f}")
    print(f"   Sex loss: {outputs.sex_loss.item():.4f}")

    # Test backward pass
    print("\n5. Testing backward pass...")
    outputs.loss.backward()
    print("   ✓ Backward pass successful")

    # Check gradient flow
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {has_grads}/{total_params}")

    # Test generation (greedy decoding)
    print("\n6. Testing generation...")
    model.eval()

    # Create a simple prompt
    prompt = torch.tensor([[tokenizer.bos_token_id]], device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_length=20,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )

    print(f"   Generated sequence: {generated[0].tolist()}")

    # Decode generated tokens
    generated_tokens = generated[0].tolist()
    category_tokens = [tid for tid in generated_tokens if tokenizer.is_category_token(tid)]
    special_tokens = [tid for tid in generated_tokens if tokenizer.is_special_token(tid)]
    code_tokens = [tid for tid in generated_tokens if tokenizer.is_code_token(tid)]

    print(f"   Category tokens: {len(category_tokens)}")
    print(f"   Special tokens: {len(special_tokens)}")
    print(f"   Code tokens: {len(code_tokens)}")

    # Decode categories
    decoded_categories = tokenizer.decode_categories(category_tokens)
    print(f"   Decoded categories: {decoded_categories[:5]}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nModel works correctly with hierarchical category-based training.")
    print("Ready for full training pipeline integration.")


if __name__ == "__main__":
    test_hierarchical_model()
