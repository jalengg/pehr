"""
ICD-9 diagnosis code hierarchy utilities.

ICD-9 codes have a natural hierarchical structure:
- Category (3 digits): "401" = Essential hypertension
- Subcategory (4th digit): "401.9" = Unspecified essential hypertension
- Specificity (5th digit): Further detail

This module provides utilities for hierarchical code generation:
1. Extract categories from specific codes
2. Build category-to-codes mappings
3. Two-stage generation: category → specific codes

Expected impact: 10-18x reduction in sparsity by learning category co-occurrence first.
"""

import logging
from collections import defaultdict
from typing import Optional
from vocabulary import DiagnosisVocabulary


class ICD9Hierarchy:
    """Manages ICD-9 diagnosis code hierarchy.

    Attributes:
        vocabulary: DiagnosisVocabulary with all specific codes.
        category_vocab: DiagnosisVocabulary with category codes only.
        category_to_codes: Dict mapping category → list of specific codes.
        code_to_category: Dict mapping specific code → category.
    """

    def __init__(
        self,
        vocabulary: DiagnosisVocabulary,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize hierarchy from vocabulary.

        Args:
            vocabulary: DiagnosisVocabulary containing all specific codes.
            logger: Optional logger for debugging.
        """
        self.vocabulary = vocabulary
        self.logger = logger

        # Build hierarchy mappings
        self.category_to_codes = defaultdict(list)
        self.code_to_category = {}

        for code in vocabulary.code2idx.keys():
            category = self.extract_category(code)
            self.category_to_codes[category].append(code)
            self.code_to_category[code] = category

        # Create category vocabulary
        categories = sorted(self.category_to_codes.keys())
        self.category_vocab = DiagnosisVocabulary()
        self.category_vocab.add_codes(categories)

        if self.logger:
            self.logger.info(f"ICD-9 Hierarchy built:")
            self.logger.info(f"  Specific codes: {len(vocabulary)}")
            self.logger.info(f"  Categories: {len(self.category_vocab)}")
            self.logger.info(f"  Avg codes per category: {len(vocabulary) / len(self.category_vocab):.1f}")
            self.logger.info(f"  Sparsity reduction: {len(vocabulary) / len(self.category_vocab):.1f}x")

    @staticmethod
    def extract_category(code: str) -> str:
        """Extract category from ICD-9 code.

        ICD-9 codes have format:
        - "401" (3 digits) → category is "401"
        - "401.9" (3 digits + decimal + 1-2 digits) → category is "401"
        - "V58.1" (V/E prefix + digits) → category is "V58"
        - "E849.0" (E prefix + digits) → category is "E849"

        Args:
            code: ICD-9 diagnosis code string.

        Returns:
            Category code (3 digits, possibly with V/E prefix).
        """
        # Handle V-codes and E-codes
        if code.startswith('V') or code.startswith('E'):
            # V58.1 → V58, E849.0 → E849
            if '.' in code:
                return code.split('.')[0]
            else:
                # V58 → V58 (already category)
                return code[:3] if len(code) >= 3 else code

        # Standard numeric codes
        if '.' in code:
            # 401.9 → 401
            return code.split('.')[0]
        else:
            # 401 → 401 (already category)
            return code[:3] if len(code) >= 3 else code

    def get_category_codes(self, category: str) -> list[str]:
        """Get all specific codes under a category.

        Args:
            category: Category code (e.g., "401").

        Returns:
            List of specific codes (e.g., ["401.1", "401.9"]).
        """
        return self.category_to_codes.get(category, [])

    def get_code_category(self, code: str) -> Optional[str]:
        """Get category for a specific code.

        Args:
            code: Specific ICD-9 code.

        Returns:
            Category code, or None if code not in vocabulary.
        """
        return self.code_to_category.get(code)

    def convert_codes_to_categories(self, codes: list[str]) -> list[str]:
        """Convert list of specific codes to categories.

        Args:
            codes: List of specific ICD-9 codes.

        Returns:
            List of category codes (duplicates removed, order preserved).
        """
        categories = []
        seen = set()

        for code in codes:
            category = self.code_to_category.get(code)
            if category and category not in seen:
                categories.append(category)
                seen.add(category)

        return categories

    def expand_categories_to_codes(
        self,
        categories: list[str],
        max_codes_per_category: int = 3
    ) -> list[str]:
        """Expand categories to specific codes (sampling).

        Args:
            categories: List of category codes.
            max_codes_per_category: Max specific codes per category.

        Returns:
            List of specific codes.
        """
        import random

        codes = []
        for category in categories:
            category_codes = self.get_category_codes(category)
            if not category_codes:
                continue

            # Sample up to max_codes_per_category
            n_sample = min(len(category_codes), max_codes_per_category)
            sampled = random.sample(category_codes, n_sample)
            codes.extend(sampled)

        return codes

    def get_hierarchy_stats(self) -> dict:
        """Get statistics about the hierarchy.

        Returns:
            Dictionary with hierarchy statistics.
        """
        codes_per_category = [len(codes) for codes in self.category_to_codes.values()]

        return {
            'n_codes': len(self.vocabulary),
            'n_categories': len(self.category_vocab),
            'sparsity_reduction': len(self.vocabulary) / len(self.category_vocab),
            'min_codes_per_category': min(codes_per_category),
            'max_codes_per_category': max(codes_per_category),
            'avg_codes_per_category': sum(codes_per_category) / len(codes_per_category),
            'median_codes_per_category': sorted(codes_per_category)[len(codes_per_category) // 2]
        }


def build_category_cooccurrence_matrix(
    training_patients: list,
    hierarchy: ICD9Hierarchy,
    logger: Optional[logging.Logger] = None
) -> tuple:
    """Build co-occurrence matrix at category level.

    This provides much better coverage than code-level co-occurrence.

    Args:
        training_patients: List of PatientRecord objects.
        hierarchy: ICD9Hierarchy instance.
        logger: Optional logger.

    Returns:
        Tuple of (cooccur_matrix, coverage_stats).
    """
    import torch
    from collections import defaultdict

    vocab_size = len(hierarchy.category_vocab)
    cooccur_counts = defaultdict(int)

    total_visits = 0
    total_pairs = 0

    if logger:
        logger.info(f"Building category co-occurrence matrix for {vocab_size} categories...")
        logger.info(f"Processing {len(training_patients)} training patients...")

    for patient in training_patients:
        for visit in patient.visits:
            if len(visit) < 2:
                continue

            total_visits += 1

            # Convert codes to categories (remove duplicates within visit)
            categories = hierarchy.convert_codes_to_categories(visit)
            category_indices = [hierarchy.category_vocab.code2idx[cat]
                              for cat in categories
                              if cat in hierarchy.category_vocab.code2idx]

            # Count pairwise co-occurrences
            for i in range(len(category_indices)):
                for j in range(i + 1, len(category_indices)):
                    idx_i = category_indices[i]
                    idx_j = category_indices[j]
                    pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                    cooccur_counts[pair] += 1
                    total_pairs += 1

    # Convert to dense symmetric matrix
    matrix = torch.zeros(vocab_size, vocab_size)
    for (i, j), count in cooccur_counts.items():
        matrix[i, j] = count
        matrix[j, i] = count

    # Compute coverage statistics
    possible_pairs = (vocab_size * (vocab_size - 1)) // 2
    unique_pairs = len(cooccur_counts)
    coverage = unique_pairs / possible_pairs if possible_pairs > 0 else 0

    stats = {
        'total_pairs': total_pairs,
        'unique_pairs': unique_pairs,
        'possible_pairs': possible_pairs,
        'coverage': coverage,
        'avg_cooccurrence': total_pairs / unique_pairs if unique_pairs > 0 else 0
    }

    if logger:
        logger.info("Category co-occurrence matrix built:")
        logger.info(f"  Total pairs observed: {total_pairs:,}")
        logger.info(f"  Unique pairs: {unique_pairs:,}")
        logger.info(f"  Possible pairs: {possible_pairs:,}")
        logger.info(f"  Coverage: {coverage*100:.2f}%")
        logger.info(f"  Average co-occurrence per pair: {stats['avg_cooccurrence']:.1f}")

    return matrix, stats


def analyze_hierarchy_improvement(
    training_patients: list,
    hierarchy: ICD9Hierarchy,
    logger: Optional[logging.Logger] = None
):
    """Analyze expected improvement from hierarchical generation.

    Compares code-level vs category-level co-occurrence coverage.

    Args:
        training_patients: List of PatientRecord objects.
        hierarchy: ICD9Hierarchy instance.
        logger: Optional logger.
    """
    from cooccurrence_utils import build_cooccurrence_matrix

    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("Analyzing Hierarchical Generation Impact")
        logger.info("=" * 80)

    # Code-level co-occurrence
    code_matrix = build_cooccurrence_matrix(
        training_patients,
        hierarchy.vocabulary,
        logger
    )

    code_pairs = (code_matrix > 0).sum().item() // 2  # Divide by 2 for symmetric
    vocab_size = len(hierarchy.vocabulary)
    code_possible = (vocab_size * (vocab_size - 1)) // 2
    code_coverage = code_pairs / code_possible if code_possible > 0 else 0

    # Category-level co-occurrence
    category_matrix, cat_stats = build_category_cooccurrence_matrix(
        training_patients,
        hierarchy,
        logger
    )

    improvement = cat_stats['coverage'] / code_coverage if code_coverage > 0 else float('inf')

    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("Comparison:")
        logger.info("=" * 80)
        logger.info(f"Code-level coverage: {code_coverage*100:.2f}%")
        logger.info(f"Category-level coverage: {cat_stats['coverage']*100:.2f}%")
        logger.info(f"Improvement factor: {improvement:.1f}x")
        logger.info("=" * 80)
