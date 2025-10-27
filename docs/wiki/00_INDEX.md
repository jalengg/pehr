# PromptEHR Documentation Wiki - Index

**Last Updated:** October 24, 2025

This wiki provides comprehensive documentation for the PromptEHR synthetic patient generation system.

## Quick Navigation

### Core Documentation

1. **[Architecture Overview](01_ARCHITECTURE.md)** - System design and component overview
2. **[Data Pipeline](02_DATA_PIPELINE.md)** - MIMIC-III preprocessing and dataset creation
3. **[Model Architecture](03_MODEL_ARCHITECTURE.md)** - Neural network design and conditioning
4. **[Training](04_TRAINING.md)** - Training loop, optimization, and metrics
5. **[Generation](05_GENERATION.md)** - Conditional and zero-prompt patient generation
6. **[Evaluation](06_EVALUATION.md)** - Medical validity and semantic coherence metrics
7. **[Usage Guide](07_USAGE_GUIDE.md)** - How to use the system
8. **[Deprecated History](08_DEPRECATED_HISTORY.md)** - Timeline of major changes

### Reference Documentation

Located in `docs/reference/`:
- **[Medical Validity Rules](../reference/medical_validity.md)** - Age/sex appropriateness constraints
- **[Multi-task Learning](../reference/multitask_learning.md)** - Auxiliary age/sex prediction
- **[Architecture Diagrams](../reference/architecture_diagrams.md)** - Visual architecture representations
- **[Architecture Explanation](../reference/architecture_explanation.md)** - Detailed component descriptions
- **[PromptEHR Comparison](../reference/promptehr_comparison.md)** - Comparison with original paper

### Historical Documentation

Located in `docs/historical/`:
- **[Tokenization Fix (2025-10-09)](../historical/2025-10-09_tokenization_fix.md)** - Early fragmentation issues
- **[Phase 1: Data Preparation](../historical/phase1_data_preparation.md)** - Vocabulary and tokenizer
- **[Phase 2: Model Architecture](../historical/phase2_model_architecture.md)** - Initial BART integration
- **[Phase 3: Dataset & Training](../historical/phase3_dataset_training.md)** - Corruption functions
- **[Phase 9: Reparameterization](../historical/phase9_reparameterization.md)** - Dual prompt conditioning
- **[Multi-task Implementation](../historical/multitask_implementation_plan.md)** - Adding auxiliary losses
- **[Conditional Reconstruction](../historical/conditional_reconstruction.md)** - Prompt-aware generation

### Analysis Documentation

Located in `docs/analysis/`:
- **[Generation Quality](../analysis/generation_quality.md)** - Quality assessment
- **[Generation Summary](../analysis/generation_summary.md)** - Generation results overview
- **[Reconstruction Analysis](../analysis/reconstruction_analysis.md)** - Reconstruction quality metrics
- **[Training Results](../analysis/training_results.md)** - Training performance logs
- **[Semantic Coherence Fix](../analysis/semantic_coherence_fix.md)** - Latest optimization attempt

## Current System Status

**Model Version:** PromptBartWithDemographicPrediction v4
**Training Data:** MIMIC-III (25,000 patients)
**Vocabulary Size:** 5,562 ICD-9 diagnosis codes

**Performance (Latest):**
- Medical Validity: 99% age-appropriate, 96% sex-appropriate, 0% duplicates
- Semantic Coherence: Needs improvement (JS divergence 0.61, top-100 overlap 0.04)
- Reconstruction Jaccard: ~0.40-0.45 (prompt-aware)

**Current Training Focus:**
- Reducing auxiliary loss weights to improve semantic coherence
- Balancing medical validity with realistic code distributions

## File Organization

```
pehr_scratch/
├── docs/
│   ├── wiki/           # This documentation
│   ├── reference/      # Reference materials
│   ├── historical/     # Implementation history
│   └── analysis/       # Quality assessments
├── deprecated/
│   ├── legacy_implementations/  # Old main.py
│   ├── unit_tests/             # Phase 1-3 tests
│   ├── utilities/              # One-off scripts
│   └── backups/                # v2, v3 directories
├── data_files/         # MIMIC-III CSVs
├── data_splits/        # Train/test split pickles
├── checkpoints/        # Local model checkpoints
├── logs/               # Training logs
├── reconstruction_results/  # Generation outputs
└── [active Python files]
```

## Getting Started

1. **New users:** Start with [Usage Guide](07_USAGE_GUIDE.md)
2. **Understanding architecture:** Read [Architecture Overview](01_ARCHITECTURE.md)
3. **Modifying training:** See [Training](04_TRAINING.md) and [Model Architecture](03_MODEL_ARCHITECTURE.md)
4. **Troubleshooting:** Check [Deprecated History](08_DEPRECATED_HISTORY.md) for known issues

## Contributing

When making changes:
1. Update relevant wiki pages
2. Add entry to root `CHANGELOG.md`
3. If deprecating code, move to `deprecated/` and document in [08_DEPRECATED_HISTORY.md](08_DEPRECATED_HISTORY.md)

## Questions?

- Check [Usage Guide](07_USAGE_GUIDE.md) for common tasks
- Review [Deprecated History](08_DEPRECATED_HISTORY.md) for why certain approaches were abandoned
- Consult reference documentation for detailed specifications
