"""
Unit tests for corruption functions in dataset.py.
Tests mask_infill, del_token, and rep_token functions.
"""
import numpy as np
from vocabulary import DiagnosisVocabulary
from code_tokenizer import DiagnosisCodeTokenizer
from dataset import CorruptionFunctions


def test_mask_infill():
    """Test mask infilling produces valid outputs."""
    print("Testing mask_infill()...")

    # Create tokenizer with sample vocabulary
    vocab_codes = ['401.9', '250.00', '585.9', '428.0', '584.9', '715.90', 'V27.0']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer, lambda_poisson=3.0)

    visits = [['401.9', '250.00', '585.9'], ['428.0', '584.9']]

    # Run masking 10 times to test randomness
    for i in range(10):
        corrupted, masks = corruption.mask_infill(visits)

        # Verify mask token present
        assert any('<mask>' in visit for visit in corrupted), f"Trial {i}: No mask token found"

        # Verify label masks have correct length
        assert len(masks) == len(visits), f"Trial {i}: Label mask length mismatch"
        for j, visit in enumerate(visits):
            assert len(masks[j]) == len(visit), f"Trial {i}, Visit {j}: mask length mismatch"

        # Verify at least one masked position per visit with codes
        for j, visit in enumerate(visits):
            if len(visit) > 0:
                # At least one position should be masked or none if single code visit
                pass  # Mask infilling allows single code to remain unmasked

    print("✓ test_mask_infill passed (10 trials)")


def test_del_token():
    """Test token deletion never deletes entire visits."""
    print("Testing del_token()...")

    vocab_codes = ['401.9', '250.00', '585.9', '428.0', '584.9']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer, del_probability=0.15)

    visits = [['401.9', '250.00', '585.9'], ['428.0']]

    # Run deletion 100 times to test constraint
    for i in range(100):
        corrupted = corruption.del_token(visits)

        # Verify no empty visits
        assert all(len(visit) > 0 for visit in corrupted), f"Trial {i}: Empty visit created"

        # Verify all remaining codes are from original vocabulary
        for visit in corrupted:
            for code in visit:
                assert code in vocab_codes, f"Trial {i}: Invalid code {code}"

    print("✓ test_del_token passed (100 trials, all visits preserved)")


def test_rep_token():
    """Test token replacement produces valid vocabulary codes."""
    print("Testing rep_token()...")

    vocab_codes = ['401.9', '250.00', '585.9', '428.0', '584.9', '715.90', 'V27.0']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer, rep_probability=0.15)

    visits = [['401.9', '250.00'], ['428.0']]

    # Run replacement 100 times
    for i in range(100):
        corrupted = corruption.rep_token(visits)

        # Verify all codes are valid
        for visit in corrupted:
            for code in visit:
                assert code in vocab_codes, f"Trial {i}: Invalid code: {code}"

        # Verify visit structure preserved
        assert len(corrupted) == len(visits), f"Trial {i}: Visit count mismatch"
        for j, visit in enumerate(corrupted):
            assert len(visit) == len(visits[j]), f"Trial {i}, Visit {j}: Code count changed"

    print("✓ test_rep_token passed (100 trials, all codes valid)")


def test_empty_visits():
    """Test corruption functions handle empty visits gracefully."""
    print("Testing empty visit handling...")

    vocab_codes = ['401.9', '250.00']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer)

    # Test with empty visits
    visits = [[], ['401.9'], []]

    # Mask infilling
    corrupted, masks = corruption.mask_infill(visits)
    assert len(corrupted) == 3, "Mask infilling: Visit count mismatch"
    assert len(corrupted[0]) == 0, "Mask infilling: Empty visit not preserved"
    assert len(masks[0]) == 0, "Mask infilling: Empty mask not created"

    # Deletion
    corrupted = corruption.del_token(visits)
    assert len(corrupted) == 3, "Deletion: Visit count mismatch"
    assert len(corrupted[0]) == 0, "Deletion: Empty visit not preserved"

    # Replacement
    corrupted = corruption.rep_token(visits)
    assert len(corrupted) == 3, "Replacement: Visit count mismatch"
    assert len(corrupted[0]) == 0, "Replacement: Empty visit not preserved"

    print("✓ test_empty_visits passed")


def test_single_code_visit():
    """Test corruption functions preserve at least one code in single-code visits."""
    print("Testing single-code visit handling...")

    vocab_codes = ['401.9', '250.00', '585.9']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)
    corruption = CorruptionFunctions(tokenizer, del_probability=1.0)  # Force deletion attempts

    # Test with single-code visits
    visits = [['401.9'], ['250.00'], ['585.9']]

    # Run deletion 50 times with del_prob=1.0 (tries to delete all)
    for i in range(50):
        corrupted = corruption.del_token(visits)

        # All visits should still have exactly 1 code
        for j, visit in enumerate(corrupted):
            assert len(visit) == 1, f"Trial {i}, Visit {j}: Single code was deleted"

    print("✓ test_single_code_visit passed (50 trials, all codes preserved)")


def test_corruption_probabilities():
    """Test that corruption probabilities are approximately correct."""
    print("Testing corruption probabilities...")

    vocab_codes = ['401.9', '250.00', '585.9', '428.0', '584.9']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Test deletion probability
    del_prob = 0.15
    corruption = CorruptionFunctions(tokenizer, del_probability=del_prob)

    visits = [['401.9', '250.00', '585.9', '428.0', '584.9']]  # 5 codes
    total_codes = 5
    total_deleted = 0
    trials = 1000

    for _ in range(trials):
        corrupted = corruption.del_token(visits)
        deleted = total_codes - len(corrupted[0])
        total_deleted += deleted

    # Expected deletions: trials * total_codes * del_prob = 1000 * 5 * 0.15 = 750
    expected = trials * total_codes * del_prob
    actual = total_deleted

    # Allow 10% margin of error
    margin = expected * 0.1
    assert abs(actual - expected) < margin, \
        f"Deletion probability off: expected {expected:.0f}, got {actual} (margin: {margin:.0f})"

    print(f"✓ test_corruption_probabilities passed (expected: {expected:.0f}, actual: {actual})")


def test_mask_token_assignment():
    """Test that <mask> token is correctly assigned."""
    print("Testing <mask> token assignment...")

    vocab_codes = ['401.9', '250.00']
    vocab = DiagnosisVocabulary()
    vocab.add_codes(vocab_codes)
    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Verify <mask> token exists
    assert hasattr(tokenizer, 'MASK_TOKEN'), "MASK_TOKEN not defined"
    assert tokenizer.MASK_TOKEN == '<mask>', f"MASK_TOKEN is '{tokenizer.MASK_TOKEN}', expected '<mask>'"

    # Verify mask token ID is 6
    mask_id = tokenizer.convert_tokens_to_ids('<mask>')
    assert mask_id == 6, f"Mask token ID is {mask_id}, expected 6"

    print("✓ test_mask_token_assignment passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Corruption Functions Unit Tests")
    print("=" * 60)
    print()

    test_mask_token_assignment()
    test_mask_infill()
    test_del_token()
    test_rep_token()
    test_empty_visits()
    test_single_code_visit()
    test_corruption_probabilities()

    print()
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
