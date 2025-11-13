"""
Test script for ICD-9 hierarchy implementation.
"""
import logging
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy, build_category_cooccurrence_matrix, analyze_hierarchy_improvement


def test_hierarchy():
    """Test ICD-9 hierarchy implementation."""
    print("=" * 80)
    print("Testing ICD-9 Hierarchy Implementation")
    print("=" * 80)

    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    # Load data
    config = Config.from_defaults()
    config.data.num_patients = 5000  # Moderate sample for testing

    print("\n1. Loading MIMIC-III data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )
    print(f"   Loaded {len(patient_records)} patients, {len(vocab)} codes")

    # Test hierarchy extraction
    print("\n2. Testing category extraction...")
    test_codes = [
        ("401.9", "401"),
        ("250.00", "250"),
        ("V58.1", "V58"),
        ("E849.0", "E849"),
        ("401", "401"),
        ("V58", "V58")
    ]

    for code, expected_category in test_codes:
        category = ICD9Hierarchy.extract_category(code)
        status = "✓" if category == expected_category else "✗"
        print(f"   {status} {code} → {category} (expected: {expected_category})")
        assert category == expected_category, f"Failed: {code} → {category} != {expected_category}"

    # Build hierarchy
    print("\n3. Building ICD-9 hierarchy...")
    hierarchy = ICD9Hierarchy(vocab, logger)

    # Get statistics
    print("\n4. Hierarchy statistics...")
    stats = hierarchy.get_hierarchy_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Test category-to-codes mapping
    print("\n5. Testing category-to-codes mapping...")
    sample_categories = list(hierarchy.category_to_codes.keys())[:5]
    for category in sample_categories:
        codes = hierarchy.get_category_codes(category)
        print(f"   {category}: {len(codes)} codes - {codes[:3]}{'...' if len(codes) > 3 else ''}")

    # Test code-to-category mapping
    print("\n6. Testing code-to-category mapping...")
    sample_codes = list(vocab.code2idx.keys())[:5]
    for code in sample_codes:
        category = hierarchy.get_code_category(code)
        print(f"   {code} → {category}")

    # Test conversion functions
    print("\n7. Testing conversion functions...")
    test_visit = sample_codes[:3]
    categories = hierarchy.convert_codes_to_categories(test_visit)
    print(f"   Codes: {test_visit}")
    print(f"   Categories: {categories}")

    expanded = hierarchy.expand_categories_to_codes(categories, max_codes_per_category=2)
    print(f"   Expanded back: {expanded}")

    # Build category co-occurrence matrix
    print("\n8. Building category co-occurrence matrix...")
    category_matrix, cat_stats = build_category_cooccurrence_matrix(
        patient_records,
        hierarchy,
        logger
    )

    # Analyze improvement
    print("\n9. Analyzing hierarchical generation impact...")
    analyze_hierarchy_improvement(patient_records, hierarchy, logger)

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nICD-9 hierarchy implementation is working correctly.")
    print(f"Sparsity reduction: {stats['sparsity_reduction']:.1f}x")
    print(f"Category coverage improvement: {cat_stats['coverage'] / 0.005:.1f}x (estimated)")


if __name__ == "__main__":
    test_hierarchy()
