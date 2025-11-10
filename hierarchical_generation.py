"""
Two-stage hierarchical generation for ICD-9 diagnosis codes.

Stage 1: Generate category sequence using trained model
Stage 2: Expand categories to specific codes via sampling
"""

import torch
import logging
import random
from typing import Optional
from hierarchical_tokenizer import HierarchicalDiagnosisTokenizer
from prompt_bart_model import PromptBartWithDemographicPrediction
from data_loader import PatientRecord


def constrain_to_category_tokens(
    logits: torch.Tensor,
    tokenizer: HierarchicalDiagnosisTokenizer
) -> torch.Tensor:
    """Constrain logits to only allow category tokens.

    Sets logits for code tokens and non-diagnostic special tokens to -inf.

    Args:
        logits: [batch, vocab_size] logits from model.
        tokenizer: HierarchicalDiagnosisTokenizer instance.

    Returns:
        Constrained logits [batch, vocab_size].
    """
    constrained_logits = logits.clone()

    # Allow special tokens (BOS, EOS, visit markers)
    # Allow category tokens
    # Block code tokens

    for token_id in range(len(tokenizer)):
        if tokenizer.is_code_token(token_id):
            # Block all specific code tokens
            constrained_logits[:, token_id] = float('-inf')
        elif tokenizer.is_special_token(token_id):
            # Allow visit start/end, EOS, but block mask and unknown
            if token_id == tokenizer.mask_token_id or token_id == tokenizer.unk_token_id:
                constrained_logits[:, token_id] = float('-inf')

    return constrained_logits


def generate_category_sequence(
    model: PromptBartWithDemographicPrediction,
    tokenizer: HierarchicalDiagnosisTokenizer,
    age: float,
    sex: int,
    device: torch.device,
    max_categories: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    logger: Optional[logging.Logger] = None
) -> list[str]:
    """Generate category sequence from demographics.

    NOTE: This is a simplified version that uses greedy generation.
    For production, this would need custom generation with logit constraints.

    Args:
        model: Trained PromptBartWithDemographicPrediction.
        tokenizer: HierarchicalDiagnosisTokenizer.
        age: Patient age (continuous).
        sex: Patient sex (0=M, 1=F).
        device: Device to run on.
        max_categories: Maximum number of categories to generate.
        temperature: Sampling temperature (not used in this simplified version).
        top_k: Top-k sampling parameter (not used).
        top_p: Top-p (nucleus) sampling parameter (not used).
        logger: Optional logger.

    Returns:
        List of category codes (e.g., ["401", "250", "V58"]).
    """
    model.eval()

    # Start with BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

    with torch.no_grad():
        # Use model's generate method (simplified greedy generation)
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_categories * 3,  # Account for visit tokens
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )

    # Extract categories from generated sequence
    generated_tokens = generated[0].tolist()
    categories = []

    for token_id in generated_tokens:
        if tokenizer.is_category_token(token_id):
            category = tokenizer.token_id_to_category[token_id]
            categories.append(category)

            if len(categories) >= max_categories:
                break

    if logger:
        logger.info(f"Generated {len(categories)} categories")

    return categories


def expand_category_to_codes(
    category: str,
    tokenizer: HierarchicalDiagnosisTokenizer,
    min_codes: int = 1,
    max_codes: int = 3,
    mode_codes: int = 2
) -> list[str]:
    """Expand single category to specific codes via sampling.

    Args:
        category: Category code (e.g., "401").
        tokenizer: HierarchicalDiagnosisTokenizer with hierarchy.
        min_codes: Minimum codes per category.
        max_codes: Maximum codes per category.
        mode_codes: Most likely number of codes (default: 2).

    Returns:
        List of specific codes (e.g., ["401.1", "401.9"]).
    """
    # Get all codes under this category
    candidate_codes = tokenizer.hierarchy.get_category_codes(category)

    if not candidate_codes:
        return []

    # Sample number of codes (triangular distribution around mode)
    n_codes = random.triangular(min_codes, max_codes, mode_codes)
    n_codes = max(min_codes, min(max_codes, int(round(n_codes))))
    n_codes = min(n_codes, len(candidate_codes))

    # Sample specific codes
    sampled_codes = random.sample(candidate_codes, n_codes)

    return sampled_codes


def expand_categories_to_codes(
    categories: list[str],
    tokenizer: HierarchicalDiagnosisTokenizer,
    logger: Optional[logging.Logger] = None
) -> list[str]:
    """Expand category sequence to specific code sequence.

    Args:
        categories: List of category codes.
        tokenizer: HierarchicalDiagnosisTokenizer with hierarchy.
        logger: Optional logger.

    Returns:
        List of specific ICD-9 codes.
    """
    all_codes = []

    for category in categories:
        codes = expand_category_to_codes(category, tokenizer)
        all_codes.extend(codes)

    if logger:
        logger.info(f"Expanded {len(categories)} categories to {len(all_codes)} codes")
        logger.info(f"Expansion ratio: {len(all_codes)/len(categories):.2f} codes/category")

    return all_codes


def generate_patient_hierarchical(
    model: PromptBartWithDemographicPrediction,
    tokenizer: HierarchicalDiagnosisTokenizer,
    age: float,
    sex: int,
    device: torch.device,
    max_categories: int = 30,
    temperature: float = 0.8,
    logger: Optional[logging.Logger] = None
) -> dict:
    """Generate synthetic patient using two-stage hierarchical generation.

    Stage 1: Generate category sequence from demographics
    Stage 2: Expand categories to specific codes

    Args:
        model: Trained model.
        tokenizer: Hierarchical tokenizer.
        age: Patient age.
        sex: Patient sex (0=M, 1=F).
        device: Device.
        max_categories: Max categories to generate.
        temperature: Sampling temperature.
        logger: Optional logger.

    Returns:
        Dictionary with:
        - categories: List of generated categories
        - codes: List of expanded specific codes
        - age: Input age
        - sex: Input sex
    """
    # Stage 1: Generate categories
    categories = generate_category_sequence(
        model=model,
        tokenizer=tokenizer,
        age=age,
        sex=sex,
        device=device,
        max_categories=max_categories,
        temperature=temperature,
        logger=logger
    )

    # Stage 2: Expand to codes
    codes = expand_categories_to_codes(
        categories=categories,
        tokenizer=tokenizer,
        logger=logger
    )

    return {
        'categories': categories,
        'codes': codes,
        'age': age,
        'sex': sex,
        'n_categories': len(categories),
        'n_codes': len(codes),
        'expansion_ratio': len(codes) / len(categories) if categories else 0
    }
