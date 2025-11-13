# 15: Training Loop

**[IN PROGRESS]**

## Key Topics

1. **Optimizer:** AdamW, lr=1e-4, weight_decay=0.01
2. **Scheduler:** Linear warmup + cosine decay
3. **Batch Processing:** DataLoader → model forward → loss backward → optimizer step
4. **Gradient Clipping:** max_norm=1.0
5. **Checkpointing:** Save every N steps

**See:** trainer.py:200-400 for main training loop

---

**Navigation:** [← 14](14_LOSS_FUNCTIONS.md) | [→ 16](16_METRICS_TRACKING.md) | [↑ Index](00_START_HERE.md)
