"""
Tokenizer for medical diagnosis codes.
Maps codes to/from integer IDs with special tokens for sequence structure.
"""
import torch
from typing import List, Dict, Optional
from vocabulary import DiagnosisVocabulary


class DiagnosisCodeTokenizer:
    """Tokenizer for diagnosis codes with special tokens for sequence structure."""

    # Special token definitions
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    V_TOKEN = "<v>"
    V_END_TOKEN = "<\\v>"
    END_TOKEN = "<END>"

    def __init__(self, vocab: DiagnosisVocabulary):
        """Initialize tokenizer with diagnosis vocabulary.

        Args:
            vocab: DiagnosisVocabulary instance containing code mappings.
        """
        self.vocab = vocab

        # Reserve first few IDs for special tokens
        self.special_token_ids = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.V_TOKEN: 3,
            self.V_END_TOKEN: 4,
            self.END_TOKEN: 5
        }

        self.id2special_token = {v: k for k, v in self.special_token_ids.items()}

        # Offset for medical codes (start after special tokens)
        self.code_offset = len(self.special_token_ids)

        # Pad token ID for PyTorch models
        self.pad_token_id = self.special_token_ids[self.PAD_TOKEN]
        self.bos_token_id = self.special_token_ids[self.BOS_TOKEN]
        self.eos_token_id = self.special_token_ids[self.EOS_TOKEN]

    def encode_codes(self, codes: List[str]) -> List[int]:
        """Encode medical codes to token IDs.

        Args:
            codes: List of diagnosis code strings.

        Returns:
            List of token IDs (offset by code_offset).
        """
        vocab_ids = self.vocab.encode(codes)
        return [idx + self.code_offset for idx in vocab_ids]

    def decode_codes(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs back to medical codes.

        Args:
            token_ids: List of token IDs.

        Returns:
            List of diagnosis code strings.
        """
        # Subtract offset to get vocabulary indices
        vocab_ids = [idx - self.code_offset for idx in token_ids if idx >= self.code_offset]
        return self.vocab.decode(vocab_ids)

    def encode_visit(self, codes: List[str], add_markers: bool = True) -> List[int]:
        """Encode a single visit (list of codes) with optional structural markers.

        Args:
            codes: List of diagnosis codes for this visit.
            add_markers: If True, wrap with <v> and <\\v> tokens.

        Returns:
            List of token IDs.
        """
        code_ids = self.encode_codes(codes)

        if add_markers:
            return [self.special_token_ids[self.V_TOKEN]] + code_ids + [self.special_token_ids[self.V_END_TOKEN]]
        else:
            return code_ids

    def encode_patient(self, visits: List[List[str]], add_special_tokens: bool = True) -> List[int]:
        """Encode full patient visit sequence.

        Args:
            visits: List of visits, where each visit is a list of codes.
            add_special_tokens: If True, add BOS/END tokens.

        Returns:
            List of token IDs representing full sequence.
        """
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.special_token_ids[self.BOS_TOKEN])

        for visit in visits:
            visit_ids = self.encode_visit(visit, add_markers=True)
            token_ids.extend(visit_ids)

        if add_special_tokens:
            token_ids.append(self.special_token_ids[self.END_TOKEN])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to human-readable string.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: If True, omit special tokens from output.

        Returns:
            Decoded string with special tokens and codes.
        """
        tokens = []

        for token_id in token_ids:
            if token_id in self.id2special_token:
                if not skip_special_tokens:
                    tokens.append(self.id2special_token[token_id])
            elif token_id >= self.code_offset:
                # Decode medical code
                vocab_idx = token_id - self.code_offset
                code = self.vocab.idx2code.get(vocab_idx, f"<UNK_{vocab_idx}>")
                tokens.append(code)

        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        """Return total vocabulary size (special tokens + medical codes)."""
        return self.code_offset + len(self.vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert special token string to ID."""
        return self.special_token_ids.get(token, -1)

    def __len__(self) -> int:
        return self.get_vocab_size()
