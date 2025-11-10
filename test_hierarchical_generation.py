"""
Test hierarchical two-stage generation.
"""
import torch
import logging
from transformers import BartConfig
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from prompt_bart_model import PromptBartWithDemographicPrediction
from hierarchical_generation import (
    constrain_to_category_tokens,
    generate_category_sequence,
    expand_category_to_codes,
    expand_categories_to_codes,
    generate_patient_hierarchical
)


def test_hierarchical_generation():
    """Test two-stage hierarchical generation."""
    print("=" * 80)
    print("Testing Hierarchical Two-Stage Generation")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    # Load data
    config = Config.from_defaults()
    config.data.num_patients = 1000

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
    print(f"   Categories: {tokenizer.get_n_categories()}")
    print(f"   Codes: {tokenizer.get_n_codes()}")

    # Initialize model (random weights for testing)
    print("\n2. Initializing model (random weights)...")
    device = torch.device('cpu')

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
    model.eval()
    print("   Model initialized")

    # Test logit constraining
    print("\n3. Testing logit constraining...")
    dummy_logits = torch.randn(1, len(tokenizer))
    constrained = constrain_to_category_tokens(dummy_logits, tokenizer)

    # Check that code tokens are blocked
    code_token_blocked = all(
        constrained[0, tid].item() == float('-inf')
        for tid in range(len(tokenizer))
        if tokenizer.is_code_token(tid)
    )
    print(f"   Code tokens blocked: {code_token_blocked}")
    assert code_token_blocked, "Code tokens should be blocked"

    # Check that category tokens are allowed
    category_token_allowed = any(
        constrained[0, tid].item() != float('-inf')
        for tid in range(len(tokenizer))
        if tokenizer.is_category_token(tid)
    )
    print(f"   Category tokens allowed: {category_token_allowed}")
    assert category_token_allowed, "Category tokens should be allowed"

    # Test category expansion
    print("\n4. Testing category expansion...")
    test_category = list(hierarchy.category_to_codes.keys())[0]
    available_codes = hierarchy.get_category_codes(test_category)
    print(f"   Test category: {test_category}")
    print(f"   Available codes: {len(available_codes)}")

    expanded = expand_category_to_codes(test_category, tokenizer)
    print(f"   Expanded to {len(expanded)} codes: {expanded}")
    assert 1 <= len(expanded) <= 3, "Should expand to 1-3 codes"
    assert all(code in available_codes for code in expanded), "All codes should be valid"

    # Test multiple category expansion
    print("\n5. Testing multiple category expansion...")
    test_categories = list(hierarchy.category_to_codes.keys())[:5]
    all_codes = expand_categories_to_codes(test_categories, tokenizer, logger)
    print(f"   Input: {len(test_categories)} categories")
    print(f"   Output: {len(all_codes)} codes")
    assert len(all_codes) >= len(test_categories), "Should have at least 1 code per category"

    # Test category sequence generation
    print("\n6. Testing category sequence generation...")
    test_age = 65.0
    test_sex = 0  # Male

    categories = generate_category_sequence(
        model=model,
        tokenizer=tokenizer,
        age=test_age,
        sex=test_sex,
        device=device,
        max_categories=10,
        temperature=1.0,
        logger=logger
    )

    print(f"   Generated {len(categories)} categories")
    print(f"   Sample categories: {categories[:5]}")

    # Verify all are valid categories
    all_valid = all(cat in hierarchy.category_vocab.code2idx for cat in categories)
    print(f"   All valid categories: {all_valid}")
    assert all_valid, "All generated tokens should be valid categories"

    # Test full patient generation
    print("\n7. Testing full patient generation...")
    result = generate_patient_hierarchical(
        model=model,
        tokenizer=tokenizer,
        age=test_age,
        sex=test_sex,
        device=device,
        max_categories=15,
        temperature=1.0,
        logger=logger
    )

    print(f"   Generated patient:")
    print(f"     Age: {result['age']}")
    print(f"     Sex: {result['sex']}")
    print(f"     Categories: {result['n_categories']}")
    print(f"     Codes: {result['n_codes']}")
    print(f"     Expansion ratio: {result['expansion_ratio']:.2f}")
    print(f"   Sample categories: {result['categories'][:5]}")
    print(f"   Sample codes: {result['codes'][:10]}")

    assert result['n_categories'] > 0, "Should generate at least one category"
    assert result['n_codes'] > 0, "Should generate at least one code"
    assert result['n_codes'] >= result['n_categories'], "Should have at least 1 code per category"

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nHierarchical two-stage generation is working correctly.")
    print("Ready for training and evaluation.")


if __name__ == "__main__":
    test_hierarchical_generation()
