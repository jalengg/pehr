"""
Vocabulary classes for medical codes.
"""
from typing import List, Dict, Optional


class DiagnosisVocabulary:
    """Vocabulary for diagnosis codes with 1:1 code-to-index mapping."""

    def __init__(self):
        self.code2idx: Dict[str, int] = {}
        self.idx2code: Dict[int, str] = {}
        self._next_idx = 0

    def add_code(self, code: str) -> int:
        """Add a diagnosis code to vocabulary, return its index."""
        if code not in self.code2idx:
            idx = self._next_idx
            self.code2idx[code] = idx
            self.idx2code[idx] = code
            self._next_idx += 1
            return idx
        return self.code2idx[code]

    def add_codes(self, codes: List[str]) -> List[int]:
        """Add multiple codes and return their indices."""
        return [self.add_code(code) for code in codes]

    def encode(self, codes: List[str]) -> List[int]:
        """Convert codes to indices."""
        return [self.code2idx[code] for code in codes]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to codes."""
        return [self.idx2code[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.code2idx)

    def __contains__(self, code: str) -> bool:
        return code in self.code2idx

    def get_vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.code2idx)
