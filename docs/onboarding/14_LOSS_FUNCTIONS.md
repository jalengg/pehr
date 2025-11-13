# 14: Loss Functions

**[IN PROGRESS]**

## Key Topics

1. **LM Loss:** CrossEntropyLoss(logits, labels), label smoothing, ignore_index=-100
2. **Age Loss:** MSELoss(age_pred, true_age), weight=0.001
3. **Sex Loss:** CrossEntropyLoss(sex_logits, true_sex), weight=0.001
4. **Co-occurrence Loss:** Penalty for rare code pairs, weight=0.05 (hierarchical only)
5. **Total Loss:** `lm_loss + 0.001*age_loss + 0.001*sex_loss + 0.05*cooccur_loss`

**See:** trainer.py, trainer_hierarchical.py for implementation

---

**Navigation:** [← 13](13_MULTI_TASK_LEARNING.md) | [→ 15](15_TRAINING_LOOP.md) | [↑ Index](00_START_HERE.md)
