"""
Metrics for evaluating PromptEHR model during training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import logging


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Perplexity measures how well the model predicts the next token.
    Lower perplexity indicates better performance.

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity value (exp(loss)).
    """
    return torch.exp(torch.tensor(loss)).item()


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Compute token-level accuracy.

    Args:
        logits: Model logits [batch, seq_len, vocab_size].
        labels: Target labels [batch, seq_len].
        ignore_index: Index to ignore (padding tokens).

    Returns:
        Accuracy as a fraction between 0 and 1.
    """
    predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]

    # Create mask for valid tokens (not padding)
    mask = labels != ignore_index

    # Count correct predictions
    correct = (predictions == labels) & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    accuracy = correct.sum().item() / total
    return accuracy


def compute_code_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    code_offset: int = 6,
    ignore_index: int = -100
) -> float:
    """Compute accuracy for diagnosis code tokens only (excluding special tokens).

    Args:
        logits: Model logits [batch, seq_len, vocab_size].
        labels: Target labels [batch, seq_len].
        code_offset: Offset where medical codes start in vocabulary (6 for special tokens).
        ignore_index: Index to ignore (padding tokens).

    Returns:
        Accuracy on medical code tokens only.
    """
    predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]

    # Mask for medical code tokens (ID >= code_offset)
    code_mask = (labels >= code_offset) & (labels != ignore_index)

    # Count correct code predictions
    correct = (predictions == labels) & code_mask
    total = code_mask.sum().item()

    if total == 0:
        return 0.0

    accuracy = correct.sum().item() / total
    return accuracy


class MetricsTracker:
    """Track metrics across training steps and epochs."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.perplexities = []
        self.token_accuracies = []
        self.code_accuracies = []
        self.lm_losses = []
        self.age_losses = []
        self.sex_losses = []
        self.reconstruction_jaccards = []

    def update(
        self,
        loss: float,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_accuracy: bool = False,
        lm_loss: Optional[float] = None,
        age_loss: Optional[float] = None,
        sex_loss: Optional[float] = None,
        reconstruction_jaccard: Optional[float] = None
    ):
        """Update metrics with new batch results.

        Args:
            loss: Batch loss value (total combined loss).
            logits: Model logits (optional, required if compute_accuracy=True).
            labels: Target labels (optional, required if compute_accuracy=True).
            compute_accuracy: Whether to compute accuracy metrics.
            lm_loss: Language modeling loss (optional).
            age_loss: Age prediction loss (optional).
            sex_loss: Sex prediction loss (optional).
            reconstruction_jaccard: Reconstruction Jaccard score (optional).
        """
        self.losses.append(loss)
        self.perplexities.append(compute_perplexity(loss))

        if lm_loss is not None:
            self.lm_losses.append(lm_loss)
        if age_loss is not None:
            self.age_losses.append(age_loss)
        if sex_loss is not None:
            self.sex_losses.append(sex_loss)
        if reconstruction_jaccard is not None:
            self.reconstruction_jaccards.append(reconstruction_jaccard)

        if compute_accuracy and logits is not None and labels is not None:
            with torch.no_grad():
                token_acc = compute_token_accuracy(logits, labels)
                code_acc = compute_code_accuracy(logits, labels)
                self.token_accuracies.append(token_acc)
                self.code_accuracies.append(code_acc)

    def get_average_metrics(self) -> dict[str, float]:
        """Get average of all tracked metrics.

        Returns:
            Dictionary with average loss, perplexity, and accuracies.
        """
        metrics = {
            'loss': sum(self.losses) / len(self.losses) if self.losses else 0.0,
            'perplexity': sum(self.perplexities) / len(self.perplexities) if self.perplexities else 0.0,
        }

        if self.lm_losses:
            metrics['lm_loss'] = sum(self.lm_losses) / len(self.lm_losses)
        if self.age_losses:
            metrics['age_loss'] = sum(self.age_losses) / len(self.age_losses)
        if self.sex_losses:
            metrics['sex_loss'] = sum(self.sex_losses) / len(self.sex_losses)
        if self.reconstruction_jaccards:
            metrics['reconstruction_jaccard'] = sum(self.reconstruction_jaccards) / len(self.reconstruction_jaccards)

        if self.token_accuracies:
            metrics['token_accuracy'] = sum(self.token_accuracies) / len(self.token_accuracies)

        if self.code_accuracies:
            metrics['code_accuracy'] = sum(self.code_accuracies) / len(self.code_accuracies)

        return metrics

    def get_last_metrics(self) -> dict[str, float]:
        """Get most recent metric values.

        Returns:
            Dictionary with last loss, perplexity, and accuracies.
        """
        metrics = {
            'loss': self.losses[-1] if self.losses else 0.0,
            'perplexity': self.perplexities[-1] if self.perplexities else 0.0,
        }

        if self.lm_losses:
            metrics['lm_loss'] = self.lm_losses[-1]
        if self.age_losses:
            metrics['age_loss'] = self.age_losses[-1]
        if self.sex_losses:
            metrics['sex_loss'] = self.sex_losses[-1]
        if self.reconstruction_jaccards:
            metrics['reconstruction_jaccard'] = self.reconstruction_jaccards[-1]

        if self.token_accuracies:
            metrics['token_accuracy'] = self.token_accuracies[-1]

        if self.code_accuracies:
            metrics['code_accuracy'] = self.code_accuracies[-1]

        return metrics


def compute_temporal_perplexity(
    model: nn.Module,
    patient_records: list,
    tokenizer,
    device: torch.device,
    logger: logging.Logger,
    max_samples: int = 500
) -> float:
    """Compute Temporal Perplexity (TPL) on patient records.

    TPL measures how well the model predicts the next visit given previous visits.
    Lower TPL indicates better temporal coherence.

    For each patient with N visits (N >= 2):
    1. Mask last visit
    2. Predict last visit from previous visits
    3. Compute cross-entropy loss
    4. Average across all predictions

    TPL = exp(average_loss)

    Args:
        model: Trained PromptBartModel.
        patient_records: List of PatientRecord objects.
        tokenizer: DiagnosisCodeTokenizer instance.
        device: Device to run on.
        logger: Logger instance.
        max_samples: Maximum number of patients to evaluate (default: 500).

    Returns:
        TPL value (lower is better).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Filter patients with 2+ visits
    multi_visit_patients = [p for p in patient_records if len(p.visits) > 1]

    if len(multi_visit_patients) == 0:
        logger.warning("No patients with 2+ visits for TPL computation")
        return float('inf')

    # Limit number of samples for faster evaluation
    if len(multi_visit_patients) > max_samples:
        indices = np.random.choice(len(multi_visit_patients), max_samples, replace=False)
        multi_visit_patients = [multi_visit_patients[i] for i in indices]

    logger.info(f"Computing TPL on {len(multi_visit_patients)} patients with 2+ visits")

    with torch.no_grad():
        for patient in multi_visit_patients:
            # Create next-visit prediction sample
            # Input: visits[0:N-1] + <mask>
            # Target: visit[N-1]

            visits = patient.visits
            num_visits = len(visits)

            # Use last visit as target
            input_visits = visits[:-1] + [[tokenizer.MASK_TOKEN]]

            # Encode input with mask
            input_ids = tokenizer.encode_patient(input_visits, add_special_tokens=False)

            # Add actual last visit for labels
            last_visit_ids = tokenizer.encode_visit(visits[-1], add_markers=True)
            input_ids.extend(last_visit_ids)

            # Add BOS and END tokens
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.convert_tokens_to_ids('<END>')]

            # Convert to tensors
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids_tensor)

            # Create labels (same as input_ids for autoregressive training)
            labels = input_ids_tensor.clone()

            # Get patient demographics
            patient_dict = patient.to_dict()
            x_num = torch.from_numpy(patient_dict['x_num']).unsqueeze(0).to(device)
            x_cat = torch.from_numpy(patient_dict['x_cat']).unsqueeze(0).to(device)

            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask,
                    labels=labels,
                    x_num=x_num,
                    x_cat=x_cat
                )

                total_loss += outputs.loss.item()
                total_samples += 1

            except RuntimeError as e:
                logger.warning(f"TPL computation failed for patient {patient.subject_id}: {e}")
                continue

    if total_samples == 0:
        logger.warning("No successful TPL samples")
        return float('inf')

    avg_loss = total_loss / total_samples
    tpl = np.exp(avg_loss)

    logger.info(f"TPL computed on {total_samples} samples: {tpl:.4f}")

    return tpl


def compute_reconstruction_jaccard(
    model: nn.Module,
    patient_records: list,
    tokenizer,
    device: torch.device,
    logger: logging.Logger,
    max_samples: int = 100,
    prompt_prob: float = 0.5
) -> float:
    """Compute prompt-aware Jaccard similarity on reconstructed codes.

    This metric measures how well the model reconstructs MASKED codes, excluding
    codes that were provided as prompts. This gives a true measure of the model's
    code generation capability independent of what it was given.

    For each patient visit:
    1. Randomly select ~50% of codes as prompts (binomial sampling)
    2. Generate remaining codes using model
    3. Compute Jaccard(generated_new, target_masked) where:
       - generated_new = generated_codes - prompt_codes
       - target_masked = target_codes - prompt_codes
    4. Average across all visits and patients

    Args:
        model: Trained PromptBartModel.
        patient_records: List of PatientRecord objects.
        tokenizer: DiagnosisCodeTokenizer instance.
        device: Device to run on.
        logger: Logger instance.
        max_samples: Maximum number of patients to evaluate (default: 100).
        prompt_prob: Probability of keeping each code as prompt (default: 0.5).

    Returns:
        Average reconstruction Jaccard score (0 to 1, higher is better).
    """
    model.eval()
    jaccard_scores = []

    # Sample patients for evaluation
    if len(patient_records) > max_samples:
        indices = np.random.choice(len(patient_records), max_samples, replace=False)
        sampled_patients = [patient_records[i] for i in indices]
    else:
        sampled_patients = patient_records

    logger.info(f"Computing reconstruction Jaccard on {len(sampled_patients)} patients")

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")

    with torch.no_grad():
        for patient in sampled_patients:
            # Get patient demographics
            patient_dict = patient.to_dict()
            x_num = torch.from_numpy(patient_dict['x_num']).unsqueeze(0).to(device)
            x_cat = torch.from_numpy(patient_dict['x_cat']).unsqueeze(0).to(device)

            # Create dummy encoder input
            encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
            encoder_attention_mask = torch.ones_like(encoder_input_ids)

            # Process each visit
            for visit_idx, target_codes in enumerate(patient.visits):
                num_codes = len(target_codes)

                if num_codes == 0:
                    continue

                # Randomly select prompt codes (binomial sampling)
                keep_mask = np.random.binomial(1, prompt_prob, num_codes).astype(bool)
                prompt_codes = [code for i, code in enumerate(target_codes) if keep_mask[i]]

                # Skip if all codes are prompts (nothing to predict)
                if len(prompt_codes) == num_codes:
                    continue

                # Encode prompt codes as decoder input
                prompt_token_ids = [tokenizer.bos_token_id, v_token_id]
                for code in prompt_codes:
                    if code in tokenizer.vocab.code2idx:
                        code_idx = tokenizer.vocab.code2idx[code]
                        code_token_id = tokenizer.code_offset + code_idx
                        prompt_token_ids.append(code_token_id)

                decoder_input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(device)

                # Generate to reconstruct full visit
                max_new_tokens = num_codes + 2

                try:
                    generated_ids = model.generate(
                        input_ids=encoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        x_num=x_num,
                        x_cat=x_cat,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        num_beams=1,
                        temperature=0.3,
                        top_k=40,
                        top_p=0.9,
                        no_repeat_ngram_size=1,
                        eos_token_id=v_end_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        bad_words_ids=[[tokenizer.bos_token_id]]
                    )

                    # Extract generated codes
                    visit_token_ids = generated_ids[0].cpu().tolist()
                    generated_code_ids = [
                        tid for tid in visit_token_ids
                        if tid >= tokenizer.code_offset
                    ]

                    # Decode codes
                    generated_codes = []
                    for tid in generated_code_ids:
                        code_idx = tid - tokenizer.code_offset
                        if code_idx < len(tokenizer.vocab):
                            code = tokenizer.vocab.idx2code[code_idx]
                            generated_codes.append(code)

                    # Compute prompt-aware Jaccard
                    # Only consider codes that were NOT in the prompt
                    generated_new = set(generated_codes) - set(prompt_codes)
                    target_masked = set(target_codes) - set(prompt_codes)

                    if len(target_masked) == 0:
                        # All target codes were prompts, skip
                        continue

                    # Jaccard similarity
                    intersection = len(generated_new & target_masked)
                    union = len(generated_new | target_masked)

                    if union > 0:
                        jaccard = intersection / union
                        jaccard_scores.append(jaccard)

                except (RuntimeError, Exception) as e:
                    logger.warning(f"Reconstruction Jaccard failed for patient {patient.subject_id}, visit {visit_idx}: {e}")
                    continue

    if len(jaccard_scores) == 0:
        logger.warning("No successful reconstruction Jaccard samples")
        return 0.0

    avg_jaccard = np.mean(jaccard_scores)
    logger.info(f"Reconstruction Jaccard computed on {len(jaccard_scores)} visits: {avg_jaccard:.4f}")

    return avg_jaccard


def compute_code_frequency_divergence(
    generated_patients: list,
    training_patients: list
) -> float:
    """Compute Jensen-Shannon divergence between code frequency distributions.

    Measures how similar the generated code frequency distribution is to the
    training distribution. Lower divergence indicates the model learned realistic
    code frequencies (common codes are common, rare codes are rare).

    Args:
        generated_patients: List of PatientRecord objects from generation.
        training_patients: List of PatientRecord objects from training data.

    Returns:
        JS divergence value (0.0-1.0, lower is better):
            < 0.1: Excellent match
            0.1-0.3: Good match
            > 0.3: Poor match
    """
    from collections import Counter
    from scipy.spatial.distance import jensenshannon

    # Count code frequencies
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_counts = Counter(gen_codes)
    train_counts = Counter(train_codes)

    # Get all unique codes
    all_codes = set(gen_counts.keys()) | set(train_counts.keys())

    # Create probability distributions (with smoothing)
    gen_probs = np.array([gen_counts.get(code, 0) + 1 for code in all_codes])
    train_probs = np.array([train_counts.get(code, 0) + 1 for code in all_codes])

    # Normalize
    gen_probs = gen_probs / gen_probs.sum()
    train_probs = train_probs / train_probs.sum()

    # Compute JS divergence
    js_div = jensenshannon(gen_probs, train_probs)

    return float(js_div)


def compute_distribution_match(
    generated_patients: list,
    training_patients: list
) -> dict:
    """Compare distributional properties using Kolmogorov-Smirnov tests.

    Tests whether generated data matches training data distributions for:
    - Number of visits per patient
    - Number of codes per visit

    Args:
        generated_patients: List of PatientRecord objects from generation.
        training_patients: List of PatientRecord objects from training data.

    Returns:
        Dictionary with p-values:
            {
                'visits_pvalue': float,
                'codes_pvalue': float,
                'visits_statistic': float,
                'codes_statistic': float
            }
        Interpretation: p-value > 0.05 means distributions match (null hypothesis)
    """
    from scipy.stats import ks_2samp

    # Extract distributions
    gen_visit_counts = [len(p.visits) for p in generated_patients]
    train_visit_counts = [len(p.visits) for p in training_patients]

    gen_codes_per_visit = [len(v) for p in generated_patients for v in p.visits]
    train_codes_per_visit = [len(v) for p in training_patients for v in p.visits]

    # Run KS tests
    visits_result = ks_2samp(gen_visit_counts, train_visit_counts)
    codes_result = ks_2samp(gen_codes_per_visit, train_codes_per_visit)

    return {
        'visits_pvalue': float(visits_result.pvalue),
        'visits_statistic': float(visits_result.statistic),
        'codes_pvalue': float(codes_result.pvalue),
        'codes_statistic': float(codes_result.statistic),
    }


def compute_top_k_overlap(
    generated_patients: list,
    training_patients: list,
    k: int = 100
) -> float:
    """Compute Jaccard similarity of top-K most common codes.

    Measures whether the model learned which codes are most common.
    High overlap indicates the model captured the frequency structure
    of the training data.

    Args:
        generated_patients: List of PatientRecord objects from generation.
        training_patients: List of PatientRecord objects from training data.
        k: Number of top codes to compare (default: 100).

    Returns:
        Jaccard similarity (0.0-1.0):
            > 0.7: Excellent overlap
            0.5-0.7: Good overlap
            < 0.5: Poor overlap
    """
    from collections import Counter

    # Count code frequencies
    gen_codes = [code for p in generated_patients for v in p.visits for code in v]
    train_codes = [code for p in training_patients for v in p.visits for code in v]

    gen_counts = Counter(gen_codes)
    train_counts = Counter(train_codes)

    # Get top-K codes
    gen_top_k = set([code for code, _ in gen_counts.most_common(k)])
    train_top_k = set([code for code, _ in train_counts.most_common(k)])

    # Compute Jaccard similarity
    intersection = len(gen_top_k & train_top_k)
    union = len(gen_top_k | train_top_k)

    if union == 0:
        return 0.0

    return intersection / union


def compute_cooccurrence_score(
    generated_patients: list,
    training_patients: list,
    logger: Optional[logging.Logger] = None
) -> float:
    """Compute average co-occurrence frequency of generated code pairs.

    Measures clinical plausibility by checking if codes that appear together
    in generated data also frequently appear together in training data.
    High scores indicate the model learned realistic code combinations.

    Args:
        generated_patients: List of PatientRecord objects from generation.
        training_patients: List of PatientRecord objects from training data.
        logger: Optional logger for progress messages.

    Returns:
        Average co-occurrence count (higher = more plausible combinations):
            > 50: Excellent (codes frequently co-occur)
            20-50: Good (codes sometimes co-occur)
            < 20: Poor (codes rarely co-occur together)
    """
    from collections import defaultdict
    from itertools import combinations

    if logger:
        logger.info("Building co-occurrence matrix from training data...")

    # Build co-occurrence matrix from training data
    cooccur_counts = defaultdict(int)
    for patient in training_patients:
        for visit in patient.visits:
            if len(visit) < 2:
                continue
            # Count all pairs in visit
            for code_a, code_b in combinations(sorted(visit), 2):
                pair = (code_a, code_b)
                cooccur_counts[pair] += 1

    if logger:
        logger.info(f"Built co-occurrence matrix with {len(cooccur_counts)} unique pairs")

    # Score generated patients
    scores = []
    for patient in generated_patients:
        for visit in patient.visits:
            if len(visit) < 2:
                continue
            # Get all pairs in generated visit
            pairs = list(combinations(sorted(visit), 2))
            # Look up co-occurrence frequency in training
            pair_scores = [cooccur_counts.get(pair, 0) for pair in pairs]
            if pair_scores:
                scores.append(np.mean(pair_scores))

    if len(scores) == 0:
        return 0.0

    avg_score = np.mean(scores)

    if logger:
        logger.info(f"Co-occurrence score computed on {len(scores)} visits: {avg_score:.2f}")

    return float(avg_score)
