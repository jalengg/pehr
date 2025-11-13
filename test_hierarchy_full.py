"""
Test ICD-9 hierarchy with full 50k patient dataset.
"""
import logging
from config import Config
from data_loader import load_mimic_data
from icd9_hierarchy import ICD9Hierarchy, analyze_hierarchy_improvement


def test_hierarchy_full():
    """Test hierarchy with full training data."""
    print("=" * 80)
    print("Testing ICD-9 Hierarchy with Full 50k Patient Dataset")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("test")

    config = Config.from_defaults()
    # Use configured 50k patients
    assert config.data.num_patients == 50000, "Config should have 50k patients"

    print(f"\nLoading {config.data.num_patients} patients...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    print(f"\nBuilding ICD-9 hierarchy...")
    hierarchy = ICD9Hierarchy(vocab, logger)

    print(f"\nHierarchy statistics:")
    stats = hierarchy.get_hierarchy_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nAnalyzing hierarchical generation impact...")
    analyze_hierarchy_improvement(patient_records, hierarchy, logger)

    print("\n" + "=" * 80)
    print("✓ Full dataset test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_hierarchy_full()
