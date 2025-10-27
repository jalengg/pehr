"""
Validation tests for Phase 3: Training Pipeline
Tests data loading, model initialization, training step, and checkpointing.
"""
import logging
import sys
import torch
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from config import Config, DataConfig, ModelConfig, TrainingConfig
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from dataset import EHRPatientDataset, EHRDataCollator
from prompt_bart_model import PromptBartModel
from metrics import compute_perplexity, compute_token_accuracy, MetricsTracker
from trainer import save_checkpoint, load_checkpoint
from transformers import BartConfig


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("test_phase3")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger


def test_config():
    """Test configuration loading."""
    logger = logging.getLogger("test_phase3")
    logger.info("=== Testing Configuration ===")

    config = Config.from_defaults()

    assert config.data.num_patients == 3000
    assert config.model.n_num_features == 1
    assert len(config.model.cat_cardinalities) == 2
    assert config.training.batch_size == 8  # Updated: reduced from 16 due to sample expansion

    logger.info("✓ Configuration loaded successfully")
    logger.debug(f"\n{config}")

    return config


def test_data_loading():
    """Test MIMIC-III data loading."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Data Loading ===")

    config = Config.from_defaults()
    config.data.num_patients = 100  # Small subset for testing

    # Load data
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    logger.debug(f"Loaded {len(patient_records)} patients")
    logger.debug(f"Vocabulary size: {len(vocab)}")

    assert len(patient_records) > 0, "No patient records loaded"
    assert len(vocab) > 0, "Vocabulary is empty"

    logger.info(f"✓ Data loading successful: {len(patient_records)} patients, {len(vocab)} codes")

    return patient_records, vocab


def test_dataset_and_collator():
    """Test EHRPatientDataset and EHRDataCollator."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Dataset and DataLoader ===")

    config = Config.from_defaults()
    config.data.num_patients = 50

    # Load data
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Create tokenizer and dataset
    tokenizer = DiagnosisCodeTokenizer(vocab)
    dataset = EHRPatientDataset(patient_records, tokenizer, logger)

    logger.debug(f"Dataset size: {len(dataset)}")

    # Test single item
    item = dataset[0]
    assert 'x_num' in item
    assert 'x_cat' in item
    assert 'token_ids' in item

    logger.debug(f"Sample item x_num shape: {item['x_num'].shape}")
    logger.debug(f"Sample item x_cat shape: {item['x_cat'].shape}")
    logger.debug(f"Sample item token_ids length: {len(item['token_ids'])}")

    # Create collator and DataLoader
    collator = EHRDataCollator(tokenizer, max_seq_length=128, logger=logger)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    # Test batch
    batch = next(iter(loader))

    # NOTE: Batch size expands due to corruption augmentation
    # batch_size=4 patients -> up to 20 samples (5 tasks per patient)
    batch_size = batch['input_ids'].shape[0]

    assert batch['x_num'].shape == (batch_size, 1)
    assert batch['x_cat'].shape == (batch_size, 2)
    assert batch['input_ids'].shape[0] == batch_size
    assert batch['attention_mask'].shape[0] == batch_size
    assert batch['labels'].shape[0] == batch_size
    assert batch_size >= 4, f"Expected at least 4 samples, got {batch_size}"

    logger.debug(f"Batch shapes (expanded from 4 patients to {batch_size} samples):")
    logger.debug(f"  x_num: {batch['x_num'].shape}")
    logger.debug(f"  x_cat: {batch['x_cat'].shape}")
    logger.debug(f"  input_ids: {batch['input_ids'].shape}")
    logger.debug(f"  attention_mask: {batch['attention_mask'].shape}")
    logger.debug(f"  labels: {batch['labels'].shape}")

    logger.info("✓ Dataset and DataLoader working correctly")

    return dataset, tokenizer


def test_model_initialization():
    """Test PromptBartModel initialization with MIMIC vocabulary."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Model Initialization ===")

    config = Config.from_defaults()
    config.data.num_patients = 50

    # Load data
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Create model
    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id

    model = PromptBartModel(
        config=bart_config,
        n_num_features=config.model.n_num_features,
        cat_cardinalities=config.model.cat_cardinalities,
        prompt_length=config.model.prompt_length
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"Model parameters: {total_params:,}")
    logger.debug(f"Model vocab size: {model.config.vocab_size}")

    assert model.config.vocab_size == len(tokenizer)
    logger.info(f"✓ Model initialized: {total_params:,} parameters, vocab_size={model.config.vocab_size}")

    return model


def test_training_step():
    """Test single training step."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Training Step ===")

    config = Config.from_defaults()
    config.data.num_patients = 50
    config.data.max_seq_length = 128

    # Load data
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    # Create dataset and DataLoader
    tokenizer = DiagnosisCodeTokenizer(vocab)
    dataset = EHRPatientDataset(patient_records, tokenizer, logger)
    collator = EHRDataCollator(tokenizer, max_seq_length=config.data.max_seq_length, logger=logger)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

    # Create model
    bart_config = BartConfig.from_pretrained(config.model.base_model)
    bart_config.vocab_size = len(tokenizer)
    bart_config.pad_token_id = tokenizer.pad_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.decoder_start_token_id = tokenizer.bos_token_id

    model = PromptBartModel(
        config=bart_config,
        n_num_features=1,
        cat_cardinalities=[2, 6],
        prompt_length=1
    )

    device = torch.device("cpu")
    model.to(device)
    model.train()

    # Get batch
    batch = next(iter(loader))

    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        labels=batch['labels'].to(device),
        x_num=batch['x_num'].to(device),
        x_cat=batch['x_cat'].to(device)
    )

    loss = outputs.loss

    logger.debug(f"Loss: {loss.item():.4f}")
    logger.debug(f"Perplexity: {compute_perplexity(loss.item()):.4f}")
    logger.debug(f"Logits shape: {outputs.logits.shape}")

    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Backward pass
    loss.backward()

    logger.info(f"✓ Training step successful: loss={loss.item():.4f}")

    return model


def test_metrics():
    """Test metrics computation."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Metrics ===")

    # Create dummy logits and labels
    batch_size, seq_len, vocab_size = 4, 20, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Add some padding
    labels[:, -5:] = -100

    # Compute metrics
    token_acc = compute_token_accuracy(logits, labels)
    logger.debug(f"Token accuracy: {token_acc:.4f}")

    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update(loss=2.5, logits=logits, labels=labels, compute_accuracy=True)
    tracker.update(loss=2.3, logits=logits, labels=labels, compute_accuracy=True)

    avg_metrics = tracker.get_average_metrics()
    logger.debug(f"Average metrics: {avg_metrics}")

    assert 'loss' in avg_metrics
    assert 'perplexity' in avg_metrics
    assert 'token_accuracy' in avg_metrics

    logger.info("✓ Metrics computation working correctly")


def test_checkpointing():
    """Test checkpoint save and load."""
    logger = logging.getLogger("test_phase3")
    logger.info("\n=== Testing Checkpointing ===")

    config = Config.from_defaults()

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create small model
        bart_config = BartConfig.from_pretrained(config.model.base_model)
        bart_config.vocab_size = 100

        model = PromptBartModel(
            config=bart_config,
            n_num_features=1,
            cat_cardinalities=[2, 6],
            prompt_length=1
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            val_loss=2.5,
            checkpoint_dir=temp_dir,
            is_best=True
        )

        checkpoint_path = Path(temp_dir) / "checkpoint_epoch_5.pt"
        best_path = Path(temp_dir) / "best_model.pt"

        assert checkpoint_path.exists(), "Checkpoint not saved"
        assert best_path.exists(), "Best model not saved"

        logger.debug(f"Checkpoint saved to {checkpoint_path}")

        # Load checkpoint
        model2 = PromptBartModel(
            config=bart_config,
            n_num_features=1,
            cat_cardinalities=[2, 6],
            prompt_length=1
        )

        epoch = load_checkpoint(str(checkpoint_path), model2)

        assert epoch == 5, f"Expected epoch 5, got {epoch}"

        logger.info("✓ Checkpointing working correctly")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all Phase 3 validation tests."""
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 3 VALIDATION: Training Pipeline")
    logger.info("=" * 80)

    try:
        # Test 1: Configuration
        config = test_config()

        # Test 2: Data loading
        patient_records, vocab = test_data_loading()

        # Test 3: Dataset and DataLoader
        dataset, tokenizer = test_dataset_and_collator()

        # Test 4: Model initialization
        model = test_model_initialization()

        # Test 5: Training step
        model = test_training_step()

        # Test 6: Metrics
        test_metrics()

        # Test 7: Checkpointing
        test_checkpointing()

        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 80)
        logger.info("\nPhase 3 Training Pipeline complete:")
        logger.info("  ✓ Configuration management")
        logger.info("  ✓ MIMIC-III data loading")
        logger.info("  ✓ Dataset and DataLoader integration")
        logger.info("  ✓ Model initialization with correct vocab")
        logger.info("  ✓ Training step (forward + backward)")
        logger.info("  ✓ Metrics computation")
        logger.info("  ✓ Checkpoint save/load")
        logger.info("\nReady to run full training with: python trainer.py")
        logger.info("Or submit to cluster with: sbatch train.slurm")

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
