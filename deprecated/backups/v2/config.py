"""
Configuration for PromptEHR training pipeline.
Centralizes all hyperparameters and paths.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    # MIMIC-III data paths
    data_dir: str = "data_files"
    patients_file: str = "PATIENTS.csv"
    admissions_file: str = "ADMISSIONS.csv"
    diagnoses_file: str = "DIAGNOSES_ICD.csv"

    # Data sampling
    num_patients: int = 3000  # Number of patients to load (None for all)
    max_seq_length: int = 256  # Maximum token sequence length
    train_val_split: float = 0.2  # Validation split ratio

    @property
    def patients_path(self) -> str:
        return str(Path(self.data_dir) / self.patients_file)

    @property
    def admissions_path(self) -> str:
        return str(Path(self.data_dir) / self.admissions_file)

    @property
    def diagnoses_path(self) -> str:
        return str(Path(self.data_dir) / self.diagnoses_file)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model
    base_model: str = "facebook/bart-base"  # Pre-trained BART model

    # Demographic conditioning
    n_num_features: int = 1  # Number of continuous features (age)
    cat_cardinalities: list[int] = None  # Category counts [n_genders, n_ethnicities]
    prompt_length: int = 1  # Number of prompt vectors per feature

    def __post_init__(self):
        if self.cat_cardinalities is None:
            self.cat_cardinalities = [2, 6]  # Gender (2), Ethnicity (6)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0  # Gradient clipping

    # Scheduler
    scheduler_type: str = "linear"  # 'linear', 'cosine', 'constant'

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision

    # Checkpointing
    checkpoint_dir: str = "/scratch/jalenj4/promptehr_checkpoints"
    save_every_n_epochs: int = 5  # Save checkpoint every N epochs
    keep_last_n_checkpoints: int = 2  # Keep only N most recent checkpoints
    save_best: bool = True  # Save best model by validation loss

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 50  # Log metrics every N steps

    # Validation
    validate_every_n_epochs: int = 1  # Run validation every N epochs

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Complete configuration combining all components."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

    @classmethod
    def from_defaults(cls) -> "Config":
        """Create configuration with default values."""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig()
        )

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("\n[Data]")
        lines.append(f"  Patients: {self.data.num_patients}")
        lines.append(f"  Max sequence length: {self.data.max_seq_length}")
        lines.append(f"  Train/val split: {self.data.train_val_split}")

        lines.append("\n[Model]")
        lines.append(f"  Base model: {self.model.base_model}")
        lines.append(f"  Continuous features: {self.model.n_num_features}")
        lines.append(f"  Categorical features: {len(self.model.cat_cardinalities)}")
        lines.append(f"  Prompt length: {self.model.prompt_length}")

        lines.append("\n[Training]")
        lines.append(f"  Batch size: {self.training.batch_size}")
        lines.append(f"  Epochs: {self.training.num_epochs}")
        lines.append(f"  Learning rate: {self.training.learning_rate}")
        lines.append(f"  Warmup steps: {self.training.warmup_steps}")
        lines.append(f"  Device: {self.training.device}")

        return "\n".join(lines)


# Default configuration instance
default_config = Config.from_defaults()
