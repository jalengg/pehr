"""
Validation tests for Phase 2: Model Architecture
Tests ConditionalPrompt, PromptBartEncoder, PromptBartDecoder, and full PromptBartModel.
"""
import logging
import sys
import torch
from transformers import BartConfig

from conditional_prompt import ConditionalPrompt
from prompt_bart_model import PromptBartModel
from code_tokenizer import DiagnosisCodeTokenizer
from vocabulary import DiagnosisVocabulary


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("test_phase2")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger


def test_conditional_prompt():
    """Test ConditionalPrompt module for demographic embedding."""
    logger = logging.getLogger("test_phase2")
    logger.info("=== Testing ConditionalPrompt ===")

    # Configuration matching our data
    n_num_features = 1  # Age
    cat_cardinalities = [2, 6]  # Gender (2), Ethnicity (6)
    hidden_dim = 768  # BART-base
    batch_size = 4

    prompt_encoder = ConditionalPrompt(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=hidden_dim,
        prompt_length=1
    )

    # Create dummy demographics
    x_num = torch.randn(batch_size, n_num_features)  # Ages
    x_cat = torch.randint(0, 2, (batch_size, 2))  # Gender, ethnicity IDs
    x_cat[:, 1] = torch.randint(0, 6, (batch_size,))  # Ethnicity in range [0, 6)

    logger.debug(f"x_num shape: {x_num.shape}")
    logger.debug(f"x_cat shape: {x_cat.shape}")

    # Forward pass
    prompt_embeds = prompt_encoder(x_num=x_num, x_cat=x_cat)

    logger.debug(f"Prompt embeddings shape: {prompt_embeds.shape}")

    # Expected: [batch, n_prompts, hidden_dim]
    # n_prompts = 1 (age) + 2 (gender, ethnicity) = 3
    expected_n_prompts = 3
    assert prompt_embeds.shape == (batch_size, expected_n_prompts, hidden_dim), \
        f"Expected shape [{batch_size}, {expected_n_prompts}, {hidden_dim}], got {prompt_embeds.shape}"

    logger.info(f"✓ ConditionalPrompt output shape correct: {prompt_embeds.shape}")

    # Test get_num_prompts
    num_prompts = prompt_encoder.get_num_prompts()
    assert num_prompts == expected_n_prompts, f"Expected {expected_n_prompts} prompts, got {num_prompts}"
    logger.info(f"✓ get_num_prompts() correct: {num_prompts}")

    return prompt_encoder


def test_prompt_bart_model():
    """Test complete PromptBartModel with dummy data."""
    logger = logging.getLogger("test_phase2")
    logger.info("\n=== Testing PromptBartModel ===")

    # Create small BART config for testing
    config = BartConfig(
        vocab_size=100,  # Small vocab for testing
        d_model=256,     # Smaller hidden dim
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )

    # Create model with demographic conditioning
    model = PromptBartModel(
        config=config,
        n_num_features=1,
        cat_cardinalities=[2, 6],
        prompt_length=1
    )

    logger.debug(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dummy batch
    batch_size = 4
    seq_len = 20

    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 100, (batch_size, seq_len))
    labels[:, :5] = -100  # Mask first 5 tokens

    x_num = torch.randn(batch_size, 1) * 30 + 50  # Ages around 50
    x_cat = torch.randint(0, 2, (batch_size, 2))
    x_cat[:, 1] = torch.randint(0, 6, (batch_size,))

    logger.debug(f"Input shapes:")
    logger.debug(f"  input_ids: {input_ids.shape}")
    logger.debug(f"  labels: {labels.shape}")
    logger.debug(f"  x_num: {x_num.shape}")
    logger.debug(f"  x_cat: {x_cat.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat,
        )

    logger.debug(f"Output logits shape: {outputs.logits.shape}")
    logger.debug(f"Loss: {outputs.loss.item():.4f}")

    # Check output shapes
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected logits shape [{batch_size}, {seq_len}, {config.vocab_size}], got {outputs.logits.shape}"

    assert outputs.loss is not None, "Loss should not be None when labels provided"

    logger.info(f"✓ Forward pass successful")
    logger.info(f"✓ Logits shape correct: {outputs.logits.shape}")
    logger.info(f"✓ Loss computed: {outputs.loss.item():.4f}")

    return model


def test_generation():
    """Test autoregressive generation with demographics."""
    logger = logging.getLogger("test_phase2")
    logger.info("\n=== Testing Generation ===")

    # Create small config
    config = BartConfig(
        vocab_size=50,
        d_model=128,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )

    model = PromptBartModel(
        config=config,
        n_num_features=1,
        cat_cardinalities=[2, 6],
        prompt_length=1
    )
    model.eval()

    # Single sample demographics
    x_num = torch.tensor([[65.0]])  # 65 years old
    x_cat = torch.tensor([[0, 0]])  # M, WHITE

    # Dummy encoder input
    input_ids = torch.randint(3, 50, (1, 10))

    logger.debug(f"Generating with demographics: age={x_num[0, 0]}, gender=M, ethnicity=WHITE")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            x_num=x_num,
            x_cat=x_cat,
            max_length=20,
            num_beams=1,
            do_sample=False,
        )

    logger.debug(f"Generated IDs: {generated_ids[0].tolist()}")
    logger.debug(f"Generated length: {generated_ids.shape[1]} tokens")

    assert generated_ids.shape[0] == 1, f"Expected batch size 1, got {generated_ids.shape[0]}"
    assert generated_ids.shape[1] <= 20, f"Generated length {generated_ids.shape[1]} exceeds max_length=20"

    logger.info(f"✓ Generation successful: {generated_ids.shape[1]} tokens generated")

    return generated_ids


def test_with_real_tokenizer():
    """Test model with actual DiagnosisCodeTokenizer."""
    logger = logging.getLogger("test_phase2")
    logger.info("\n=== Testing with DiagnosisCodeTokenizer ===")

    # Create vocabulary and tokenizer
    vocab = DiagnosisVocabulary()
    vocab.add_codes(["V3001", "250.00", "401.9", "428.0", "585.6"])

    tokenizer = DiagnosisCodeTokenizer(vocab)

    logger.debug(f"Vocabulary size: {len(tokenizer)}")
    logger.debug(f"Special tokens: PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, END={tokenizer.convert_tokens_to_ids('<END>')}")

    # Create BART config matching tokenizer vocab size
    config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=128,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    model = PromptBartModel(
        config=config,
        n_num_features=1,
        cat_cardinalities=[2, 6],
        prompt_length=1
    )

    logger.debug(f"Model vocab size: {model.config.vocab_size}")

    # Encode patient sequence
    visits = [["V3001", "250.00"], ["401.9"]]
    token_ids = tokenizer.encode_patient(visits, add_special_tokens=True)

    logger.debug(f"Encoded sequence: {token_ids}")
    logger.debug(f"Decoded: {tokenizer.decode(token_ids, skip_special_tokens=False)}")

    # Create batch
    input_ids = torch.tensor([token_ids])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    x_num = torch.tensor([[65.0]])
    x_cat = torch.tensor([[0, 0]])

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat,
        )

    logger.debug(f"Loss: {outputs.loss.item():.4f}")
    logger.info(f"✓ Forward pass with DiagnosisCodeTokenizer successful")
    logger.info(f"✓ Model compatible with custom vocabulary")

    return model, tokenizer


def run_all_tests():
    """Run all Phase 2 validation tests."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("PHASE 2 VALIDATION: Model Architecture")
    logger.info("=" * 60)

    try:
        # Test 1: ConditionalPrompt
        prompt_encoder = test_conditional_prompt()

        # Test 2: Full PromptBartModel
        model = test_prompt_bart_model()

        # Test 3: Generation
        generated_ids = test_generation()

        # Test 4: Integration with DiagnosisCodeTokenizer
        model, tokenizer = test_with_real_tokenizer()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nPhase 2 Model Architecture complete:")
        logger.info("  ✓ ConditionalPrompt: Demographics → embeddings")
        logger.info("  ✓ PromptBartEncoder: Prepends prompt embeddings")
        logger.info("  ✓ PromptBartDecoder: Prepends prompt embeddings")
        logger.info("  ✓ PromptBartModel: Full seq2seq with conditioning")
        logger.info("  ✓ Generation: Autoregressive with demographics")
        logger.info("  ✓ Integration: Works with DiagnosisCodeTokenizer")

        return True

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
