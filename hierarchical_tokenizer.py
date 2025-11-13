"""
Hierarchical tokenizer for ICD-9 diagnosis codes.

Supports two-stage generation:
1. Generate category sequence (e.g., "401", "250", "585")
2. Expand categories to specific codes (e.g., "401.9", "250.00", "585.9")

Token vocabulary structure:
- Special tokens: <s>, <pad>, </s>, <unk>, <v>, <\v>, <mask> (IDs 0-6)
- Category tokens: IDs 7 to 7+n_categories-1
- Code tokens: IDs 7+n_categories to 7+n_categories+n_codes-1

This allows the model to learn category-level patterns first (better coverage),
then refine to specific codes.
"""

from typing import Optional
from icd9_hierarchy import ICD9Hierarchy


class HierarchicalDiagnosisTokenizer:
    """Tokenizer supporting hierarchical ICD-9 code generation.

    Attributes:
        hierarchy: ICD9Hierarchy managing category-code mappings.
        special_tokens: List of special token strings.
        pad_token_id: Padding token ID.
        bos_token_id: Beginning of sequence token ID.
        eos_token_id: End of sequence token ID.
        unk_token_id: Unknown token ID.
        mask_token_id: Mask token ID.
        visit_start_token_id: Visit start token ID.
        visit_end_token_id: Visit end token ID.
        code_offset: Token ID where category tokens start.
        category_offset: Token ID where category tokens start (same as code_offset).
        specific_code_offset: Token ID where specific code tokens start.
    """

    def __init__(self, hierarchy: ICD9Hierarchy):
        """Initialize hierarchical tokenizer.

        Args:
            hierarchy: ICD9Hierarchy with category-code mappings.
        """
        self.hierarchy = hierarchy

        # Special tokens (IDs 0-6)
        self.special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<v>', '<\\v>', '<mask>']
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.visit_start_token_id = 4
        self.visit_end_token_id = 5
        self.mask_token_id = 6

        # Category tokens start at ID 7
        self.code_offset = 7
        self.category_offset = 7

        # Build category token mapping
        self.category_to_token_id = {}
        self.token_id_to_category = {}

        for category, idx in hierarchy.category_vocab.code2idx.items():
            token_id = self.category_offset + idx
            self.category_to_token_id[category] = token_id
            self.token_id_to_category[token_id] = category

        # Specific code tokens start after category tokens
        n_categories = len(hierarchy.category_vocab)
        self.specific_code_offset = self.category_offset + n_categories

        # Build specific code token mapping
        self.code_to_token_id = {}
        self.token_id_to_code = {}

        for code, idx in hierarchy.vocabulary.code2idx.items():
            token_id = self.specific_code_offset + idx
            self.code_to_token_id[code] = token_id
            self.token_id_to_code[token_id] = code

        self.vocab_size = self.specific_code_offset + len(hierarchy.vocabulary)

    def encode_categories(self, categories: list[str]) -> list[int]:
        """Encode category sequence to token IDs.

        Args:
            categories: List of category codes.

        Returns:
            List of token IDs.
        """
        return [self.category_to_token_id.get(cat, self.unk_token_id)
                for cat in categories]

    def encode_codes(self, codes: list[str]) -> list[int]:
        """Encode specific code sequence to token IDs.

        Args:
            codes: List of specific ICD-9 codes.

        Returns:
            List of token IDs.
        """
        return [self.code_to_token_id.get(code, self.unk_token_id)
                for code in codes]

    def decode_categories(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs to category codes.

        Args:
            token_ids: List of token IDs.

        Returns:
            List of category codes.
        """
        categories = []
        for token_id in token_ids:
            if token_id in self.token_id_to_category:
                categories.append(self.token_id_to_category[token_id])
        return categories

    def decode_codes(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs to specific codes.

        Args:
            token_ids: List of token IDs.

        Returns:
            List of specific ICD-9 codes.
        """
        codes = []
        for token_id in token_ids:
            if token_id in self.token_id_to_code:
                codes.append(self.token_id_to_code[token_id])
        return codes

    def encode_visit_as_categories(self, visit: list[str]) -> list[int]:
        """Encode visit codes as category tokens.

        Args:
            visit: List of specific codes in visit.

        Returns:
            List of category token IDs (duplicates removed).
        """
        categories = self.hierarchy.convert_codes_to_categories(visit)
        return self.encode_categories(categories)

    def encode_patient_as_categories(self, visits: list[list[str]]) -> list[int]:
        """Encode full patient record as category token sequence.

        Format: <s> <v> cat1 cat2 <\\v> <v> cat3 <\\v> </s>

        Args:
            visits: List of visits, each containing specific codes.

        Returns:
            List of token IDs (category-based sequence).
        """
        tokens = [self.bos_token_id]

        for visit in visits:
            tokens.append(self.visit_start_token_id)
            category_tokens = self.encode_visit_as_categories(visit)
            tokens.extend(category_tokens)
            tokens.append(self.visit_end_token_id)

        tokens.append(self.eos_token_id)
        return tokens

    def is_category_token(self, token_id: int) -> bool:
        """Check if token ID corresponds to a category.

        Args:
            token_id: Token ID to check.

        Returns:
            True if category token, False otherwise.
        """
        return self.category_offset <= token_id < self.specific_code_offset

    def is_code_token(self, token_id: int) -> bool:
        """Check if token ID corresponds to a specific code.

        Args:
            token_id: Token ID to check.

        Returns:
            True if code token, False otherwise.
        """
        return token_id >= self.specific_code_offset

    def is_special_token(self, token_id: int) -> bool:
        """Check if token ID is a special token.

        Args:
            token_id: Token ID to check.

        Returns:
            True if special token, False otherwise.
        """
        return token_id < self.code_offset

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def get_n_categories(self) -> int:
        """Return number of category tokens."""
        return len(self.hierarchy.category_vocab)

    def get_n_codes(self) -> int:
        """Return number of specific code tokens."""
        return len(self.hierarchy.vocabulary)
