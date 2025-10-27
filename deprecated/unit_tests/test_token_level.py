"""
Test token-level age prediction architecture.
"""
import torch
from transformers import BartConfig

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from prompt_bart_model import PromptBartWithDemographicPrediction


def test_age_to_class():
    """Test age to class conversion."""
    print("Testing age_to_class conversion...")

    config = BartConfig(
        vocab_size=100,
        d_model=768,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
    )

    model_config = Config.from_defaults()
    model = PromptBartWithDemographicPrediction(
        config=config,
        n_num_features=1,
        cat_cardinalities=[2],
        age_loss_weight=0.05,
        sex_loss_weight=0.2
    )

    # Test age conversions
    ages = torch.tensor([0.5, 1.5, 5.0, 15.0, 25.0, 50.0, 75.0, 90.0])
    expected_classes = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5])

    classes = model.age_to_class(ages)

    print(f"Ages: {ages.tolist()}")
    print(f"Expected classes: {expected_classes.tolist()}")
    print(f"Actual classes: {classes.tolist()}")

    assert torch.all(classes == expected_classes), "Age classification failed!"
    print("✓ Age to class conversion works correctly\n")


def test_forward_pass():
    """Test forward pass with token-level age prediction."""
    print("Testing forward pass with token-level age prediction...")

    # Load small sample of data
    config = Config.from_defaults()
    import logging
    test_logger = logging.getLogger("test")
    test_logger.setLevel(logging.WARNING)  # Suppress info logs

    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        num_patients=10,
        logger=test_logger
    )

    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Create model
    bart_config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=768,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = PromptBartWithDemographicPrediction(
        config=bart_config,
        n_num_features=1,
        cat_cardinalities=[2],
        age_loss_weight=0.05,
        sex_loss_weight=0.2
    )

    model.train()

    # Create dummy batch
    batch_size = 2
    seq_len = 20

    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    labels = torch.randint(tokenizer.code_offset, len(tokenizer), (batch_size, seq_len))
    labels[:, :5] = -100  # Mask first 5 tokens (special tokens)

    x_num = torch.tensor([[25.0], [75.0]])  # Young adult and elderly
    x_cat = torch.tensor([[0], [1]])  # M and F

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        x_num=x_num,
        x_cat=x_cat,
        code_offset=tokenizer.code_offset
    )

    print(f"Total loss: {outputs.loss.item():.4f}")
    print(f"LM loss: {outputs.lm_loss.item():.4f}")
    print(f"Age loss (CE): {outputs.age_loss.item():.4f}")
    print(f"Sex loss: {outputs.sex_loss.item():.4f}")

    # Verify losses are reasonable
    assert outputs.loss.item() > 0, "Total loss should be positive"
    assert outputs.lm_loss.item() > 0, "LM loss should be positive"
    assert outputs.age_loss.item() >= 0, "Age loss should be non-negative"
    assert outputs.sex_loss.item() > 0, "Sex loss should be positive"

    # Verify age loss is classification (bounded by log(6) ≈ 1.79 for random)
    assert outputs.age_loss.item() < 3.0, "Age loss too high for classification"

    print("✓ Forward pass works correctly\n")


def test_age_loss_weight():
    """Verify age loss weight is reduced correctly."""
    print("Testing age loss weighting...")

    config = Config.from_defaults()
    assert config.model.age_loss_weight == 0.05, f"Expected 0.05, got {config.model.age_loss_weight}"
    print(f"✓ Age loss weight correctly set to {config.model.age_loss_weight}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Token-Level Age Prediction Architecture Tests")
    print("=" * 80)
    print()

    test_age_to_class()
    test_age_loss_weight()
    test_forward_pass()

    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
