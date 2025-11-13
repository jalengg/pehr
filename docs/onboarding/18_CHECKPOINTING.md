# 18: Checkpointing

**[IN PROGRESS]**

## Key Topics

1. **Save Frequency:** Every 500 steps, end of epoch
2. **Checkpoint Contents:** model_state_dict, optimizer_state_dict, epoch, step
3. **Best Model Tracking:** Save best model by validation loss
4. **Resume Training:** Load checkpoint and continue

**See:** trainer.py:500-600 for checkpoint logic

---

**Navigation:** [← 17](17_HYPERPARAMETERS.md) | [→ 19](19_GENERATION_MODES.md) | [↑ Index](00_START_HERE.md)
