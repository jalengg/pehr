"""
Co-occurrence utilities for semantic coherence improvement.

This module provides tools to:
1. Build co-occurrence matrices from training data
2. Compute co-occurrence regularization loss during training
3. Analyze code co-occurrence patterns

Novel contribution: PromptEHR does not implement co-occurrence-aware training.
This module enables explicit learning of code co-occurrence patterns.
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Optional


def build_cooccurrence_matrix(
    training_patients: list,
    vocabulary,
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """
    Build pairwise code co-occurrence matrix from training data.

    This matrix M[i,j] stores how many times code i and code j appear
    together in the same visit across all training patients.

    Args:
        training_patients: List of PatientRecord objects
        vocabulary: DiagnosisCodeVocabulary instance
        logger: Optional logger for progress updates

    Returns:
        Symmetric matrix [vocab_size, vocab_size] with co-occurrence counts

    Example:
        >>> matrix = build_cooccurrence_matrix(train_patients, vocab)
        >>> # matrix[42, 157] = number of visits containing both code 42 and 157
    """
    vocab_size = len(vocabulary)

    if logger:
        logger.info(f"Building co-occurrence matrix for {vocab_size} codes...")
        logger.info(f"Processing {len(training_patients)} training patients...")

    # Use defaultdict for efficient sparse matrix construction
    cooccur_counts = defaultdict(int)
    total_pairs = 0

    for patient_idx, patient in enumerate(training_patients):
        for visit in patient.visits:
            if len(visit) < 2:
                # Single-code visits have no co-occurrences
                continue

            # Convert codes to indices
            code_indices = []
            for code in visit:
                if code in vocabulary.code2idx:
                    code_indices.append(vocabulary.code2idx[code])

            # Count all pairwise co-occurrences in this visit
            for i in range(len(code_indices)):
                for j in range(i + 1, len(code_indices)):
                    code_i = code_indices[i]
                    code_j = code_indices[j]

                    # Use tuple for symmetric access
                    pair = (min(code_i, code_j), max(code_i, code_j))
                    cooccur_counts[pair] += 1
                    total_pairs += 1

        # Progress logging
        if logger and (patient_idx + 1) % 5000 == 0:
            logger.info(f"  Processed {patient_idx + 1}/{len(training_patients)} patients...")

    # Convert sparse dict to dense tensor
    matrix = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)

    for (i, j), count in cooccur_counts.items():
        matrix[i, j] = count
        matrix[j, i] = count  # Symmetric

    if logger:
        unique_pairs = len(cooccur_counts)
        possible_pairs = vocab_size * (vocab_size - 1) // 2
        coverage = unique_pairs / possible_pairs * 100

        logger.info(f"Co-occurrence matrix built:")
        logger.info(f"  Total code pairs observed: {total_pairs:,}")
        logger.info(f"  Unique code pairs: {unique_pairs:,}")
        logger.info(f"  Possible pairs: {possible_pairs:,}")
        logger.info(f"  Coverage: {coverage:.2f}%")
        logger.info(f"  Average co-occurrence per observed pair: {total_pairs / unique_pairs:.1f}")

    return matrix


def save_cooccurrence_matrix(matrix: torch.Tensor, filepath: str):
    """
    Save co-occurrence matrix to disk.

    Args:
        matrix: Co-occurrence matrix tensor
        filepath: Path to save file (.pt format)
    """
    torch.save(matrix, filepath)


def load_cooccurrence_matrix(filepath: str, device: torch.device = None) -> torch.Tensor:
    """
    Load co-occurrence matrix from disk.

    Args:
        filepath: Path to saved matrix file
        device: Device to load matrix on (default: CPU)

    Returns:
        Co-occurrence matrix tensor
    """
    if device is None:
        device = torch.device('cpu')

    return torch.load(filepath, map_location=device)


def cooccurrence_loss(
    generated_code_ids: torch.Tensor,
    cooccur_matrix: torch.Tensor,
    tokenizer,
    threshold: int = 5,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute co-occurrence regularization loss for generated codes.

    Penalizes generating code pairs that rarely co-occur in training data.
    Loss is higher for pairs with lower training co-occurrence frequency.

    Args:
        generated_code_ids: Token IDs of generated codes [batch, seq_len]
        cooccur_matrix: Pre-computed co-occurrence matrix [vocab_size, vocab_size]
        tokenizer: DiagnosisCodeTokenizer (to identify code tokens)
        threshold: Minimum co-occurrence count to not penalize (default: 5)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss (if reduction='mean' or 'sum') or per-sample loss [batch]

    Example:
        >>> # During training
        >>> outputs = model(input_ids, labels=labels, x_num=x_num, x_cat=x_cat)
        >>> lm_loss = outputs.loss
        >>> cooccur_loss = cooccurrence_loss(input_ids, cooccur_matrix, tokenizer)
        >>> total_loss = lm_loss + 0.05 * cooccur_loss
    """
    batch_size, seq_len = generated_code_ids.shape
    device = generated_code_ids.device

    # Move matrix to same device as inputs
    if cooccur_matrix.device != device:
        cooccur_matrix = cooccur_matrix.to(device)

    batch_losses = []

    for batch_idx in range(batch_size):
        sequence = generated_code_ids[batch_idx]

        # Extract only code token IDs (skip special tokens)
        code_ids = []
        for token_id in sequence:
            token_id_val = token_id.item()
            # Check if this is a code token (>= code_offset)
            if token_id_val >= tokenizer.code_offset:
                code_idx = token_id_val - tokenizer.code_offset
                if code_idx < len(tokenizer.vocab):
                    code_ids.append(code_idx)

        if len(code_ids) < 2:
            # No co-occurrences in this sequence
            batch_losses.append(torch.tensor(0.0, device=device))
            continue

        # Compute pairwise co-occurrence penalties
        sample_loss = 0.0
        num_pairs = 0

        for i in range(len(code_ids)):
            for j in range(i + 1, len(code_ids)):
                code_i = code_ids[i]
                code_j = code_ids[j]

                # Look up co-occurrence frequency
                cooccur_count = cooccur_matrix[code_i, code_j].item()

                # Penalize if rare co-occurrence
                if cooccur_count < threshold:
                    # Penalty inversely proportional to frequency
                    # penalty = threshold / (count + 1)
                    # Example: count=0 → penalty=5, count=2 → penalty=1.67, count=4 → penalty=1
                    penalty = threshold / (cooccur_count + 1.0)
                    sample_loss += penalty

                num_pairs += 1

        # Average over pairs in this sample
        if num_pairs > 0:
            sample_loss = sample_loss / num_pairs

        batch_losses.append(torch.tensor(sample_loss, device=device))

    # Stack batch losses
    batch_losses = torch.stack(batch_losses)

    # Apply reduction
    if reduction == 'mean':
        return batch_losses.mean()
    elif reduction == 'sum':
        return batch_losses.sum()
    else:  # 'none'
        return batch_losses


def cooccurrence_loss_efficient(
    generated_code_ids: torch.Tensor,
    cooccur_matrix: torch.Tensor,
    tokenizer,
    threshold: int = 5
) -> torch.Tensor:
    """
    Efficient vectorized version of co-occurrence loss.

    This version processes entire batch at once using tensor operations.
    Significantly faster for large batches.

    Args:
        generated_code_ids: Token IDs [batch, seq_len]
        cooccur_matrix: Co-occurrence matrix [vocab_size, vocab_size]
        tokenizer: DiagnosisCodeTokenizer
        threshold: Minimum co-occurrence count

    Returns:
        Scalar loss (mean over batch)
    """
    batch_size, seq_len = generated_code_ids.shape
    device = generated_code_ids.device

    if cooccur_matrix.device != device:
        cooccur_matrix = cooccur_matrix.to(device)

    # Create mask for code tokens (not special tokens)
    code_mask = generated_code_ids >= tokenizer.code_offset

    # Convert to code indices (subtract offset)
    code_indices = generated_code_ids - tokenizer.code_offset

    # Get vocab size (handle both DiagnosisCodeTokenizer and HierarchicalDiagnosisTokenizer)
    if hasattr(tokenizer, 'vocab'):
        vocab_size = len(tokenizer.vocab)
    elif hasattr(tokenizer, 'hierarchy'):
        vocab_size = len(tokenizer.hierarchy.vocabulary)
    else:
        vocab_size = cooccur_matrix.shape[0]

    code_indices = code_indices.clamp(min=0, max=vocab_size - 1)

    # Zero out non-code positions
    code_indices = code_indices * code_mask.long()

    total_loss = 0.0
    total_pairs = 0

    for batch_idx in range(batch_size):
        # Get valid code indices for this sample
        valid_codes = code_indices[batch_idx][code_mask[batch_idx]]

        if len(valid_codes) < 2:
            continue

        # Build pairwise indices
        num_codes = len(valid_codes)
        pairs_i, pairs_j = torch.triu_indices(num_codes, num_codes, offset=1, device=device)

        # Get code indices for pairs
        codes_i = valid_codes[pairs_i]
        codes_j = valid_codes[pairs_j]

        # Look up co-occurrence counts
        cooccur_counts = cooccur_matrix[codes_i, codes_j]

        # Compute penalties (only for rare pairs)
        rare_mask = cooccur_counts < threshold
        penalties = threshold / (cooccur_counts + 1.0)
        penalties = penalties * rare_mask.float()

        total_loss += penalties.sum()
        total_pairs += len(pairs_i)

    # Average over all pairs in batch
    if total_pairs > 0:
        return total_loss / total_pairs
    else:
        return torch.tensor(0.0, device=device)


def analyze_cooccurrence_patterns(
    generated_patients: list,
    training_patients: list,
    vocabulary,
    logger: logging.Logger,
    top_k: int = 20
):
    """
    Analyze and report co-occurrence patterns in generated vs training data.

    Useful for debugging and understanding what the model learned.

    Args:
        generated_patients: List of generated PatientRecord objects
        training_patients: List of training PatientRecord objects
        vocabulary: DiagnosisCodeVocabulary instance
        logger: Logger for output
        top_k: Number of top pairs to report

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 80)
    logger.info("Co-occurrence Pattern Analysis")
    logger.info("=" * 80)

    # Build matrices
    logger.info("\nBuilding co-occurrence matrices...")
    gen_matrix = build_cooccurrence_matrix(generated_patients, vocabulary)
    train_matrix = build_cooccurrence_matrix(training_patients, vocabulary)

    # Find most common pairs in training
    train_pairs = []
    for i in range(len(vocabulary)):
        for j in range(i + 1, len(vocabulary)):
            count = train_matrix[i, j].item()
            if count > 0:
                train_pairs.append((i, j, count))

    train_pairs.sort(key=lambda x: x[2], reverse=True)

    # Find most common pairs in generated
    gen_pairs = []
    for i in range(len(vocabulary)):
        for j in range(i + 1, len(vocabulary)):
            count = gen_matrix[i, j].item()
            if count > 0:
                gen_pairs.append((i, j, count))

    gen_pairs.sort(key=lambda x: x[2], reverse=True)

    # Report top pairs
    logger.info(f"\nTop {top_k} Code Pairs in Training Data:")
    for rank, (i, j, count) in enumerate(train_pairs[:top_k], 1):
        code_i = vocabulary.idx2code[i]
        code_j = vocabulary.idx2code[j]
        logger.info(f"  {rank:2d}. ({code_i}, {code_j}): {count:4.0f} co-occurrences")

    logger.info(f"\nTop {top_k} Code Pairs in Generated Data:")
    for rank, (i, j, count) in enumerate(gen_pairs[:top_k], 1):
        code_i = vocabulary.idx2code[i]
        code_j = vocabulary.idx2code[j]

        # Check training frequency
        train_count = train_matrix[i, j].item()

        logger.info(f"  {rank:2d}. ({code_i}, {code_j}): {count:4.0f} gen, {train_count:4.0f} train")

    # Compute overlap metrics
    train_pair_set = {(i, j) for i, j, _ in train_pairs}
    gen_pair_set = {(i, j) for i, j, _ in gen_pairs}

    intersection = train_pair_set & gen_pair_set
    union = train_pair_set | gen_pair_set

    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

    logger.info(f"\nPair-Level Statistics:")
    logger.info(f"  Unique pairs in training: {len(train_pair_set):,}")
    logger.info(f"  Unique pairs in generated: {len(gen_pair_set):,}")
    logger.info(f"  Overlapping pairs: {len(intersection):,}")
    logger.info(f"  Jaccard similarity: {jaccard:.4f}")

    # Frequency correlation
    common_pairs = list(intersection)
    if len(common_pairs) > 0:
        train_freqs = [train_matrix[i, j].item() for i, j in common_pairs]
        gen_freqs = [gen_matrix[i, j].item() for i, j in common_pairs]

        # Spearman correlation
        from scipy.stats import spearmanr
        corr, pvalue = spearmanr(train_freqs, gen_freqs)

        logger.info(f"  Frequency correlation (Spearman): {corr:.4f} (p={pvalue:.4f})")

    logger.info("=" * 80)

    return {
        'train_unique_pairs': len(train_pair_set),
        'gen_unique_pairs': len(gen_pair_set),
        'overlapping_pairs': len(intersection),
        'jaccard_similarity': jaccard
    }


def get_highly_cooccurring_codes(
    code: str,
    cooccur_matrix: torch.Tensor,
    vocabulary,
    top_k: int = 10
) -> list:
    """
    Get codes that frequently co-occur with a given code.

    Useful for understanding learned patterns and for data augmentation.

    Args:
        code: ICD-9 code string (e.g., "401.9")
        cooccur_matrix: Co-occurrence matrix
        vocabulary: DiagnosisCodeVocabulary
        top_k: Number of top co-occurring codes to return

    Returns:
        List of (code, frequency) tuples, sorted by frequency

    Example:
        >>> neighbors = get_highly_cooccurring_codes("401.9", matrix, vocab, top_k=5)
        >>> # [("428.0", 150), ("250.00", 120), ...]
    """
    if code not in vocabulary.code2idx:
        return []

    code_idx = vocabulary.code2idx[code]

    # Get co-occurrence frequencies with all other codes
    frequencies = cooccur_matrix[code_idx].cpu().numpy()

    # Get top-k indices (excluding self)
    top_indices = np.argsort(frequencies)[::-1]
    top_indices = [i for i in top_indices if i != code_idx][:top_k]

    # Convert to codes
    result = []
    for idx in top_indices:
        neighbor_code = vocabulary.idx2code[idx]
        freq = frequencies[idx]
        if freq > 0:  # Only include non-zero
            result.append((neighbor_code, freq))

    return result
