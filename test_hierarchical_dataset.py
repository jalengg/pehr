"""
Test hierarchical dataset implementation.
"""
import logging
import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from hierarchical_dataset import HierarchicalEHRDataset, HierarchicalEHRDataCollator


def test_hierarchical_dataset():
    """Test hierarchical dataset and data collator."""
    print("=" * 80)
    print("Testing Hierarchical EHR Dataset")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    # Load data
    config = Config.from_defaults()
    config.data.num_patients = 100

    print("\n1. Loading data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Build hierarchy and tokenizer
    print("\n2. Building hierarchy and tokenizer...")
    hierarchy = ICD9Hierarchy(vocab, logger=None)
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Categories: {tokenizer.get_n_categories()}")
    print(f"   Codes: {tokenizer.get_n_codes()}")

    # Create dataset
    print("\n3. Creating hierarchical dataset...")
    dataset = HierarchicalEHRDataset(patient_records, tokenizer, logger)
    print(f"   Dataset size: {len(dataset)}")

    # Test single item
    print("\n4. Testing single item retrieval...")
    sample = dataset[0]
    print(f"   Keys: {list(sample.keys())}")
    print(f"   x_num shape: {sample['x_num'].shape}")
    print(f"   x_cat shape: {sample['x_cat'].shape}")
    print(f"   Token IDs shape: {sample['token_ids'].shape}")
    print(f"   Token IDs (first 15): {sample['token_ids'][:15]}")

    # Verify token types
    category_count = sum(1 for tid in sample['token_ids'] if tokenizer.is_category_token(tid))
    special_count = sum(1 for tid in sample['token_ids'] if tokenizer.is_special_token(tid))
    print(f"   Category tokens: {category_count}")
    print(f"   Special tokens: {special_count}")

    # Create data collator
    print("\n5. Creating data collator...")
    collator = HierarchicalEHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        logger=logger,
        lambda_poisson=config.training.lambda_poisson,
        corruption_prob=0.5,
        use_mask_infilling=True
    )
    print("   Collator initialized")

    # Test batching without corruption
    print("\n6. Testing batch collation (no corruption)...")
    collator.corruption_prob = 0.0
    batch = collator([dataset[i] for i in range(4)])
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   x_num shape: {batch['x_num'].shape}")
    print(f"   x_cat shape: {batch['x_cat'].shape}")

    # Verify labels match input_ids (no corruption)
    labels_match = (batch['input_ids'] == batch['labels']).sum().item()
    total_non_padding = (batch['labels'] != -100).sum().item()
    print(f"   Labels matching input_ids: {labels_match}/{total_non_padding} "
          f"({100*labels_match/total_non_padding:.1f}%)")

    # Test batching with corruption
    print("\n7. Testing batch collation (with corruption)...")
    collator.corruption_prob = 1.0
    batch_corrupted = collator([dataset[i] for i in range(4)])

    # Check for mask tokens in input
    mask_count = (batch_corrupted['input_ids'] == tokenizer.mask_token_id).sum().item()
    print(f"   Mask tokens in input: {mask_count}")

    # Check that labels have -100 for non-masked positions
    non_ignored = (batch_corrupted['labels'] != -100).sum().item()
    print(f"   Non-ignored positions in labels: {non_ignored}")

    # Test DataLoader
    print("\n8. Testing DataLoader integration...")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator
    )
    print(f"   DataLoader batches: {len(dataloader)}")

    # Iterate through one batch
    batch = next(iter(dataloader))
    print(f"   Loaded batch with {batch['input_ids'].shape[0]} samples")
    print(f"   Sequence length: {batch['input_ids'].shape[1]}")

    # Verify tensor dtypes
    assert batch['input_ids'].dtype == torch.int64, "input_ids should be int64"
    assert batch['labels'].dtype == torch.int64, "labels should be int64"
    assert batch['x_num'].dtype == torch.float32, "x_num should be float32"
    assert batch['x_cat'].dtype == torch.int64, "x_cat should be int64"

    # Verify no code tokens in input (should be all categories)
    has_code_tokens = False
    for token_id in batch['input_ids'].flatten():
        if tokenizer.is_code_token(token_id.item()):
            has_code_tokens = True
            break

    assert not has_code_tokens, "Input should only contain category tokens, not specific codes"

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nHierarchical dataset is working correctly.")
    print("Ready for model training on category sequences.")


if __name__ == "__main__":
    test_hierarchical_dataset()
