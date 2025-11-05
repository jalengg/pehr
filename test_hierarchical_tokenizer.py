"""
Test hierarchical tokenizer implementation.
"""
import logging
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer


def test_hierarchical_tokenizer():
    """Test hierarchical tokenizer."""
    print("=" * 80)
    print("Testing Hierarchical Diagnosis Tokenizer")
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
    print(f"   Codes: {len(vocab)}, Categories: {len(hierarchy.category_vocab)}")

    # Create tokenizer
    print("\n2. Creating hierarchical tokenizer...")
    tokenizer = HierarchicalDiagnosisTokenizer(hierarchy)
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Category tokens: {tokenizer.category_offset} to {tokenizer.specific_code_offset-1}")
    print(f"   Code tokens: {tokenizer.specific_code_offset} to {len(tokenizer)-1}")

    # Test category encoding/decoding
    print("\n3. Testing category encoding/decoding...")
    test_categories = ["401", "250", "V58"]
    category_tokens = tokenizer.encode_categories(test_categories)
    decoded_categories = tokenizer.decode_categories(category_tokens)
    print(f"   Input: {test_categories}")
    print(f"   Tokens: {category_tokens}")
    print(f"   Decoded: {decoded_categories}")
    assert decoded_categories == test_categories, "Category encoding/decoding failed"

    # Test code encoding/decoding
    print("\n4. Testing code encoding/decoding...")
    test_codes = ["401.9", "250.00", "V58.1"]
    # Get actual codes from vocab
    actual_codes = list(vocab.code2idx.keys())[:3]
    code_tokens = tokenizer.encode_codes(actual_codes)
    decoded_codes = tokenizer.decode_codes(code_tokens)
    print(f"   Input: {actual_codes}")
    print(f"   Tokens: {code_tokens}")
    print(f"   Decoded: {decoded_codes}")
    assert decoded_codes == actual_codes, "Code encoding/decoding failed"

    # Test visit encoding as categories
    print("\n5. Testing visit encoding as categories...")
    sample_patient = patient_records[0]
    sample_visit = sample_patient.visits[0]
    print(f"   Original visit codes: {sample_visit[:5]}...")
    category_tokens = tokenizer.encode_visit_as_categories(sample_visit)
    decoded_categories = tokenizer.decode_categories(category_tokens)
    print(f"   Category tokens: {category_tokens[:5]}...")
    print(f"   Decoded categories: {decoded_categories[:5]}...")
    print(f"   Compression: {len(sample_visit)} codes → {len(decoded_categories)} categories")

    # Test full patient encoding
    print("\n6. Testing full patient encoding as categories...")
    patient_tokens = tokenizer.encode_patient_as_categories(sample_patient.visits)
    print(f"   Patient token sequence length: {len(patient_tokens)}")
    print(f"   First 15 tokens: {patient_tokens[:15]}")

    # Verify structure
    assert patient_tokens[0] == tokenizer.bos_token_id, "Should start with BOS"
    assert patient_tokens[-1] == tokenizer.eos_token_id, "Should end with EOS"
    assert tokenizer.visit_start_token_id in patient_tokens, "Should contain visit start"
    assert tokenizer.visit_end_token_id in patient_tokens, "Should contain visit end"

    # Test token type checking
    print("\n7. Testing token type checking...")
    category_token = category_tokens[0]
    code_token = code_tokens[0]
    special_token = tokenizer.bos_token_id

    print(f"   Token {category_token} is_category: {tokenizer.is_category_token(category_token)}")
    print(f"   Token {code_token} is_code: {tokenizer.is_code_token(code_token)}")
    print(f"   Token {special_token} is_special: {tokenizer.is_special_token(special_token)}")

    assert tokenizer.is_category_token(category_token), "Should be category token"
    assert tokenizer.is_code_token(code_token), "Should be code token"
    assert tokenizer.is_special_token(special_token), "Should be special token"

    # Test tokenizer statistics
    print("\n8. Tokenizer statistics...")
    print(f"   Total vocab size: {len(tokenizer)}")
    print(f"   Special tokens: {len(tokenizer.special_tokens)}")
    print(f"   Category tokens: {tokenizer.get_n_categories()}")
    print(f"   Code tokens: {tokenizer.get_n_codes()}")
    print(f"   Expected vocab size: {7 + tokenizer.get_n_categories() + tokenizer.get_n_codes()}")

    expected_size = 7 + tokenizer.get_n_categories() + tokenizer.get_n_codes()
    assert len(tokenizer) == expected_size, f"Vocab size mismatch: {len(tokenizer)} != {expected_size}"

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nHierarchical tokenizer is working correctly.")


if __name__ == "__main__":
    test_hierarchical_tokenizer()
