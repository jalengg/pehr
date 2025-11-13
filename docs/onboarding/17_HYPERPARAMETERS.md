# 17: Hyperparameters

**[IN PROGRESS]**

## Key Parameters (config.py)

- **Model:** vocab_size=6992, d_model=768, layers=6-6, heads=12
- **Training:** batch_size=32, lr=1e-4, epochs=50, warmup_steps=1000
- **Data:** max_seq_len=512, corruption_prob=0.5, lambda_poisson=3.0
- **Loss weights:** age=0.001, sex=0.001, cooccur=0.05

**See:** config.py for full configuration

---

**Navigation:** [← 16](16_METRICS_TRACKING.md) | [→ 18](18_CHECKPOINTING.md) | [↑ Index](00_START_HERE.md)
