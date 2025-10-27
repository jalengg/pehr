"""
Validation tests for Phase 1: Data Preparation
Tests vocabulary, tokenizer, and data loading without fragmentation.
"""
import logging
import sys
from vocabulary import DiagnosisVocabulary
from code_tokenizer import DiagnosisCodeTokenizer
from data_loader import PatientRecord, ETHNICITY_CATEGORIES


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("test_phase1")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger


def test_vocabulary():
    """Test DiagnosisVocabulary for 1:1 code-to-ID mapping."""
    logger = logging.getLogger("test_phase1")
    logger.info("=== Testing Vocabulary ===")

    vocab = DiagnosisVocabulary()

    # Test adding codes
    test_codes = ["V3001", "250.00", "401.9", "428.0", "V3001"]  # Duplicate V3001

    for code in test_codes:
        idx = vocab.add_code(code)
        logger.debug(f"Added code '{code}' -> ID {idx}")

    # Should have 4 unique codes (V3001 appears twice)
    assert len(vocab) == 4, f"Expected 4 unique codes, got {len(vocab)}"
    logger.info(f"✓ Vocabulary size correct: {len(vocab)}")

    # Test encoding
    codes_to_encode = ["V3001", "250.00", "401.9"]
    encoded = vocab.encode(codes_to_encode)
    logger.debug(f"Encoded {codes_to_encode} -> {encoded}")

    # Test decoding
    decoded = vocab.decode(encoded)
    logger.debug(f"Decoded {encoded} -> {decoded}")

    assert decoded == codes_to_encode, f"Decode failed: {decoded} != {codes_to_encode}"
    logger.info(f"✓ Encoding/decoding roundtrip successful")

    # Test no fragmentation - each code is single ID
    assert len(encoded) == len(codes_to_encode), "Codes were fragmented!"
    logger.info(f"✓ No fragmentation: {len(codes_to_encode)} codes -> {len(encoded)} IDs")

    return vocab


def test_tokenizer(vocab):
    """Test DiagnosisCodeTokenizer for sequence encoding."""
    logger = logging.getLogger("test_phase1")
    logger.info("\n=== Testing Tokenizer ===")

    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Test special tokens
    logger.debug(f"Special tokens:")
    logger.debug(f"  PAD: {tokenizer.pad_token_id}")
    logger.debug(f"  BOS: {tokenizer.bos_token_id}")
    logger.debug(f"  <v>: {tokenizer.convert_tokens_to_ids('<v>')}")
    logger.debug(f"  <\\v>: {tokenizer.convert_tokens_to_ids('<\\v>')}")
    logger.debug(f"  <END>: {tokenizer.convert_tokens_to_ids('<END>')}")

    # Test visit encoding
    visit_codes = ["V3001", "250.00"]
    visit_ids = tokenizer.encode_visit(visit_codes, add_markers=True)
    logger.debug(f"Visit {visit_codes} -> {visit_ids}")

    # Should be: [<v>, code1, code2, <\v>]
    assert len(visit_ids) == 4, f"Expected 4 tokens (markers + 2 codes), got {len(visit_ids)}"
    logger.info(f"✓ Visit encoding correct: {len(visit_codes)} codes -> {len(visit_ids)} tokens (with markers)")

    # Test patient sequence encoding
    visits = [
        ["V3001", "250.00"],
        ["401.9"]
    ]
    patient_ids = tokenizer.encode_patient(visits, add_special_tokens=True)
    logger.debug(f"Patient visits {visits} -> {patient_ids}")

    # Should be: [BOS, <v>, code1, code2, <\v>, <v>, code3, <\v>, END]
    # = 1 + (1 + 2 + 1) + (1 + 1 + 1) + 1 = 9 tokens
    expected_len = 1 + 4 + 3 + 1  # BOS + visit1 + visit2 + END
    assert len(patient_ids) == expected_len, f"Expected {expected_len} tokens, got {len(patient_ids)}"
    logger.info(f"✓ Patient sequence encoding correct: {expected_len} tokens")

    # Test decoding
    decoded_str = tokenizer.decode(patient_ids, skip_special_tokens=False)
    logger.debug(f"Decoded: {decoded_str}")

    # Verify codes are intact (not fragmented)
    assert "V3001" in decoded_str, "V3001 code missing in decoded output"
    assert "250.00" in decoded_str, "250.00 code missing in decoded output"
    assert "401.9" in decoded_str, "401.9 code missing in decoded output"
    logger.info(f"✓ Decoding preserves intact codes (no fragmentation)")

    # Test vocabulary size
    vocab_size = tokenizer.get_vocab_size()
    logger.debug(f"Total vocab size: {vocab_size} (6 special + {len(vocab)} codes)")
    assert vocab_size == tokenizer.code_offset + len(vocab), "Vocab size mismatch"
    logger.info(f"✓ Vocabulary size correct: {vocab_size}")

    return tokenizer


def test_patient_record():
    """Test PatientRecord data structure."""
    logger = logging.getLogger("test_phase1")
    logger.info("\n=== Testing PatientRecord ===")

    # Create sample patient (ethnicity removed)
    record = PatientRecord(
        subject_id=12345,
        age=65.0,
        gender="M",
        visits=[
            ["V3001", "V053"],
            ["250.00", "401.9"]
        ]
    )

    logger.debug(f"Created patient: ID={record.subject_id}, age={record.age}, gender={record.gender}")

    # Convert to dict
    patient_dict = record.to_dict()

    # Check x_num (age)
    assert patient_dict['x_num'].shape == (1,), f"x_num shape wrong: {patient_dict['x_num'].shape}"
    assert patient_dict['x_num'][0] == 65.0, f"Age incorrect: {patient_dict['x_num'][0]}"
    logger.info(f"✓ x_num (age) correct: {patient_dict['x_num']}")

    # Check x_cat (gender only - race removed)
    assert patient_dict['x_cat'].shape == (1,), f"x_cat shape wrong: {patient_dict['x_cat'].shape}"
    gender_id = patient_dict['x_cat'][0]

    assert gender_id == 0, f"Gender encoding wrong: M should be 0, got {gender_id}"  # M=0, F=1
    logger.info(f"✓ x_cat (gender) correct: {patient_dict['x_cat']}")

    # Check visits
    assert patient_dict['num_visits'] == 2, f"Visit count wrong: {patient_dict['num_visits']}"
    assert len(patient_dict['visits']) == 2, f"Visit list wrong: {len(patient_dict['visits'])}"
    logger.info(f"✓ Visits structure correct: {patient_dict['num_visits']} visits")

    return record


def test_no_fragmentation():
    """Critical test: Verify medical codes are NOT fragmented."""
    logger = logging.getLogger("test_phase1")
    logger.info("\n=== Testing NO FRAGMENTATION ===")

    # Test challenging codes with decimals, letters, etc.
    challenging_codes = [
        "V3001",       # Letter + digits
        "250.00",      # Decimal
        "401.9",       # Decimal with single fraction
        "E950.4",      # Letter + decimal
        "99.04",       # Double digit + decimal
        "V053"         # Short code
    ]

    vocab = DiagnosisVocabulary()
    vocab.add_codes(challenging_codes)

    logger.debug(f"Testing {len(challenging_codes)} challenging codes")

    # Encode all codes
    encoded = vocab.encode(challenging_codes)

    logger.info(f"Input: {len(challenging_codes)} codes")
    logger.info(f"Encoded: {len(encoded)} token IDs")

    # CRITICAL: Number of codes should equal number of token IDs (1:1 mapping)
    assert len(encoded) == len(challenging_codes), \
        f"FRAGMENTATION DETECTED: {len(challenging_codes)} codes -> {len(encoded)} tokens"

    # Decode and verify exact match
    decoded = vocab.decode(encoded)
    for original, decoded_code in zip(challenging_codes, decoded):
        assert original == decoded_code, \
            f"Code mismatch: '{original}' != '{decoded_code}'"
        logger.debug(f"  ✓ {original} -> {vocab.code2idx[original]} -> {decoded_code}")

    logger.info(f"✓✓✓ NO FRAGMENTATION: All {len(challenging_codes)} codes preserved intact")


def run_all_tests():
    """Run all Phase 1 validation tests."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("PHASE 1 VALIDATION: Data Preparation")
    logger.info("=" * 60)

    try:
        # Test 1: Vocabulary
        vocab = test_vocabulary()

        # Test 2: Tokenizer
        tokenizer = test_tokenizer(vocab)

        # Test 3: Patient Record
        record = test_patient_record()

        # Test 4: No Fragmentation (critical)
        test_no_fragmentation()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 1 Data Preparation complete:")
        logger.info("  ✓ Vocabulary: 1:1 code-to-ID mapping")
        logger.info("  ✓ Tokenizer: Structural tokens + code tokens")
        logger.info("  ✓ Patient records: Demographics separated from codes")
        logger.info("  ✓ NO FRAGMENTATION: Medical codes preserved intact")

        return True

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
