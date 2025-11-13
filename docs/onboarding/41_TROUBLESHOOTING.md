# 41: Troubleshooting

**[IN PROGRESS]**

## Common Issues

1. **CUDA OOM:** Reduce batch_size in config.py
2. **Tokenizer mismatch:** Check vocab_size matches tokenizer
3. **Label masking:** Ensure padding tokens set to -100
4. **NaN losses:** Check for divide-by-zero in loss computation

**See:** [08_DEPRECATED_HISTORY.md](../wiki/08_DEPRECATED_HISTORY.md) for historical issues and solutions

---

**Navigation:** [↑ Index](00_START_HERE.md)
