# 10: Data Flow Integration - End-to-End Pipeline

**Estimated Time:** 45 minutes
**Prerequisites:** [08_DATA_LOADING.md](08_DATA_LOADING.md), [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)
**Next:** [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md)

---

## [IN PROGRESS]

This page will cover the complete data pipeline from MIMIC-III CSVs to model input tensors.

### Key Topics to Cover

1. **Pipeline Overview**
   - CSV → PatientRecord → DiagnosisVocabulary
   - PatientRecord → EHRPatientDataset → DataLoader
   - Batch processing with EHRDataCollator

2. **Data Flow Diagram**
   ```
   MIMIC-III CSVs
        ↓
   load_mimic_data()
        ↓
   List[PatientRecord] + DiagnosisVocabulary
        ↓
   DiagnosisCodeTokenizer
        ↓
   EHRPatientDataset (corruption + tokenization)
        ↓
   DataLoader + EHRDataCollator (batching)
        ↓
   Batch tensors (input_ids, labels, x_num, x_cat)
        ↓
   Model forward pass
   ```

3. **Code Example**
   ```python
   # Load data
   patients, vocab = load_mimic_data(...)

   # Create tokenizer
   tokenizer = DiagnosisCodeTokenizer(vocab)

   # Create dataset
   dataset = EHRPatientDataset(patients, tokenizer, corruption_prob=0.5)

   # Create dataloader
   collator = EHRDataCollator(tokenizer)
   dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

   # Get batch
   batch = next(iter(dataloader))
   # batch = {
   #     'input_ids': tensor [batch_size, seq_len],
   #     'labels': tensor [batch_size, seq_len],
   #     'x_num': tensor [batch_size, 1],
   #     'x_cat': tensor [batch_size, 1]
   # }
   ```

4. **Tensor Shapes**
   - input_ids: [batch_size, max_seq_len] - Corrupted token IDs
   - labels: [batch_size, max_seq_len] - Original token IDs (with -100 for padding)
   - x_num: [batch_size, 1] - Age (float32)
   - x_cat: [batch_size, 1] - Gender ID (int64)

5. **Memory Considerations**
   - Full dataset (46K patients): ~2GB RAM
   - Training batch (32 patients): ~50MB GPU memory
   - Max sequence length: 512 tokens (truncation if longer)

**See files:**
- `data_loader.py` - Loading pipeline
- `dataset.py` - Dataset and collator
- `trainer.py` - Integration with training loop

---

## What's Next?

**Next:** [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md) - BART encoder-decoder architecture, embeddings, attention

---

**Navigation:**
- ← Back to [09_DATASET_CORRUPTION.md](09_DATASET_CORRUPTION.md)
- → Next: [11_MODEL_ARCHITECTURE.md](11_MODEL_ARCHITECTURE.md)
- ↑ Up to [00_START_HERE.md](00_START_HERE.md)
