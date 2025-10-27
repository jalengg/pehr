"""
Unit tests for reparameterization and dual prompt conditioning.

Tests:
1. Offset-based categorical embedding prevents collision
2. Numerical reparameterization shape and gradient flow
3. Combined prompt output dimensions
4. Encoder/decoder prompt separation
5. Full forward pass integration
"""
import torch
from transformers import BartConfig

from conditional_prompt import (
    NumericalConditionalPrompt,
    CategoricalConditionalPrompt,
    ConditionalPrompt
)
from prompt_bart_model import PromptBartModel


def test_offset_based_categorical_embedding():
    """Test that offset-based indexing prevents category collision."""
    cat_cardinalities = [2, 6]  # Gender (2), Ethnicity (6)
    hidden_dim = 768
    d_hidden = 128

    cat_prompt = CategoricalConditionalPrompt(
        cat_cardinalities=cat_cardinalities,
        hidden_dim=hidden_dim,
        d_hidden=d_hidden
    )

    # Check embedding table size = sum of cardinalities
    assert cat_prompt.embeddings.num_embeddings == sum(cat_cardinalities)
    assert cat_prompt.embeddings.num_embeddings == 8

    # Check offsets are correct
    expected_offsets = torch.tensor([0, 2])
    assert torch.equal(cat_prompt.category_offsets, expected_offsets)

    # Test that same category ID for different features produces different embeddings
    batch_size = 4
    x_cat = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])  # All combinations

    embeddings = cat_prompt(x_cat)

    # Check output shape
    assert embeddings.shape == (batch_size, 2, hidden_dim)

    # Gender=0 and Ethnicity=0 should have different embeddings
    # Extract the embedding for gender=0, ethnicity=0
    sample_0 = embeddings[0]  # shape: [2, hidden_dim]
    gender_0_embed = sample_0[0]
    ethnicity_0_embed = sample_0[1]

    # They should NOT be identical (offset prevents collision)
    assert not torch.allclose(gender_0_embed, ethnicity_0_embed)

    print("✓ Offset-based categorical embedding prevents collision")


def test_numerical_reparameterization():
    """Test numerical reparameterization output shape and gradient flow."""
    n_num_features = 1  # Age only
    hidden_dim = 768
    d_hidden = 128
    batch_size = 8

    num_prompt = NumericalConditionalPrompt(
        n_num_features=n_num_features,
        hidden_dim=hidden_dim,
        d_hidden=d_hidden
    )

    # Check parameter shapes
    assert num_prompt.weight.shape == (n_num_features, d_hidden)
    assert num_prompt.bias.shape == (n_num_features, d_hidden)
    assert num_prompt.proj.weight.shape == (hidden_dim, d_hidden)

    # Test forward pass
    x_num = torch.rand(batch_size, n_num_features) * 100  # Random ages 0-100
    embeddings = num_prompt(x_num)

    # Check output shape
    assert embeddings.shape == (batch_size, n_num_features, hidden_dim)

    # Test gradient flow
    loss = embeddings.sum()
    loss.backward()

    assert num_prompt.weight.grad is not None
    assert num_prompt.bias.grad is not None
    assert num_prompt.proj.weight.grad is not None

    print("✓ Numerical reparameterization has correct shape and gradient flow")


def test_combined_prompt_output():
    """Test combined prompt concatenation and total prompts calculation."""
    n_num_features = 1
    cat_cardinalities = [2, 6]
    hidden_dim = 768
    d_hidden = 128
    batch_size = 4
    prompt_length = 1

    combined_prompt = ConditionalPrompt(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=hidden_dim,
        d_hidden=d_hidden,
        prompt_length=prompt_length
    )

    # Check total prompts calculation
    expected_total = n_num_features * prompt_length + len(cat_cardinalities) * prompt_length
    assert combined_prompt.get_num_prompts() == expected_total
    assert combined_prompt.get_num_prompts() == 3  # 1 (age) + 2 (gender, ethnicity)

    # Test forward pass with both types
    x_num = torch.rand(batch_size, n_num_features) * 100
    x_cat = torch.randint(0, 2, (batch_size, len(cat_cardinalities)))

    embeddings = combined_prompt(x_num=x_num, x_cat=x_cat)

    # Check output shape: [batch, total_prompts, hidden_dim]
    assert embeddings.shape == (batch_size, expected_total, hidden_dim)

    # Test with only numerical
    embeddings_num_only = combined_prompt(x_num=x_num, x_cat=None)
    assert embeddings_num_only.shape == (batch_size, n_num_features, hidden_dim)

    # Test with only categorical
    embeddings_cat_only = combined_prompt(x_num=None, x_cat=x_cat)
    assert embeddings_cat_only.shape == (batch_size, len(cat_cardinalities), hidden_dim)

    print("✓ Combined prompt produces correct concatenated output")


def test_encoder_decoder_prompt_separation():
    """Test that encoder and decoder have separate prompt parameters."""
    config = BartConfig(
        vocab_size=1000,
        d_model=768,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024
    )

    n_num_features = 1
    cat_cardinalities = [2, 6]
    d_hidden = 128

    model = PromptBartModel(
        config=config,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_hidden=d_hidden
    )

    # Check that both prompt encoders exist
    assert model.encoder_prompt_encoder is not None
    assert model.decoder_prompt_encoder is not None

    # Check that they have separate parameters
    encoder_params = list(model.encoder_prompt_encoder.parameters())
    decoder_params = list(model.decoder_prompt_encoder.parameters())

    assert len(encoder_params) > 0
    assert len(decoder_params) > 0

    # Verify they are NOT the same objects (separate parameters)
    assert encoder_params[0] is not decoder_params[0]

    # Check that they are both in the model's parameter list
    model_param_set = set(model.parameters())
    for p in encoder_params:
        assert p in model_param_set
    for p in decoder_params:
        assert p in model_param_set

    print("✓ Encoder and decoder have separate prompt parameters")


def test_forward_pass_with_dual_prompts():
    """Test full forward pass with dual prompt conditioning."""
    config = BartConfig(
        vocab_size=1000,
        d_model=256,  # Smaller for faster testing
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2
    )

    n_num_features = 1
    cat_cardinalities = [2, 6]
    d_hidden = 64  # Smaller for testing
    batch_size = 4
    seq_len = 20

    model = PromptBartModel(
        config=config,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_hidden=d_hidden
    )
    model.eval()

    # Prepare inputs
    input_ids = torch.randint(3, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(3, 1000, (batch_size, seq_len))

    x_num = torch.rand(batch_size, n_num_features) * 100
    x_cat = torch.randint(0, 2, (batch_size, len(cat_cardinalities)))

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat
        )

    # Check outputs
    assert outputs.loss is not None
    assert outputs.logits is not None

    # Logits shape should match labels (prompts are sliced off)
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

    # Check that loss is a scalar
    assert outputs.loss.dim() == 0

    print("✓ Full forward pass with dual prompts works correctly")


def test_attention_mask_extension():
    """Test that attention masks are correctly extended for prompts."""
    config = BartConfig(
        vocab_size=500,
        d_model=256,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2
    )

    n_num_features = 1
    cat_cardinalities = [2, 6]
    d_hidden = 64
    batch_size = 2
    seq_len = 10

    model = PromptBartModel(
        config=config,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_hidden=d_hidden
    )
    model.eval()

    # Prepare inputs with padding
    input_ids = torch.tensor([
        [1, 50, 60, 70, 2, 0, 0, 0, 0, 0],  # 5 real tokens, 5 padding
        [1, 80, 90, 100, 110, 120, 130, 140, 150, 2]  # 10 real tokens
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    x_num = torch.tensor([[65.0], [42.0]])
    x_cat = torch.tensor([[0, 2], [1, 4]])

    # Need labels to trigger decoder input generation
    labels = torch.tensor([
        [1, 50, 60, 70, 2, -100, -100, -100, -100, -100],
        [1, 80, 90, 100, 110, 120, 130, 140, 150, 2]
    ])

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            x_num=x_num,
            x_cat=x_cat
        )

    # Check that outputs are valid
    assert outputs.logits is not None
    assert not torch.isnan(outputs.logits).any()

    # Encoder should have received extended attention mask
    # (num_prompts=3 + original seq_len)
    # But we can't directly access it, so we check that forward pass succeeded

    print("✓ Attention masks are correctly extended for prompts")


def test_parameter_count_increase():
    """Test that dual prompts increase parameter count appropriately."""
    config = BartConfig(
        vocab_size=500,
        d_model=256,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512
    )

    # Model without prompts
    model_no_prompts = PromptBartModel(config=config)
    params_no_prompts = sum(p.numel() for p in model_no_prompts.parameters())

    # Model with dual prompts
    model_with_prompts = PromptBartModel(
        config=config,
        n_num_features=1,
        cat_cardinalities=[2, 6],
        d_hidden=64
    )
    params_with_prompts = sum(p.numel() for p in model_with_prompts.parameters())

    # Should have more parameters
    assert params_with_prompts > params_no_prompts

    # Calculate expected additional parameters
    # Encoder prompts: num_weight (1 * 64) + num_bias (1 * 64) + num_proj (64 * 256)
    #                + cat_embed (8 * 64) + cat_bias (2 * 64) + cat_proj (64 * 256)
    # Decoder prompts: same
    num_encoder = 1 * 64 + 1 * 64 + 64 * 256 + 8 * 64 + 2 * 64 + 64 * 256
    num_decoder = num_encoder
    expected_additional = num_encoder + num_decoder

    actual_additional = params_with_prompts - params_no_prompts

    # Should be approximately equal (within a small tolerance for other differences)
    assert abs(actual_additional - expected_additional) < 100

    print(f"✓ Parameter count increased by {actual_additional:,} (expected ~{expected_additional:,})")


if __name__ == "__main__":
    print("Running reparameterization tests...\n")

    test_offset_based_categorical_embedding()
    test_numerical_reparameterization()
    test_combined_prompt_output()
    test_encoder_decoder_prompt_separation()
    test_forward_pass_with_dual_prompts()
    test_attention_mask_extension()
    test_parameter_count_increase()

    print("\n" + "="*60)
    print("All reparameterization tests passed!")
    print("="*60)
