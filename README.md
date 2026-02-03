# Copilot Instructions for Finetune-LLM

## Project Overview
This project finetunes GPT-2 on code diffs to generate commit messages. The model learns the relationship between code changes and their corresponding commit messages using the CommitBench dataset (10% subset).

## Core Architecture

### Data Pipeline
- **Input**: CSV files containing `diff` and `message` columns
- **Processing Flow**: CSV → Cleaned text → Tokenized sequences → PyTorch tensors
- **Key files**: [train_llm.py](train_llm.py) (lines 30-56 handle preprocessing)

### Training Loop Architecture
The training uses a standard supervised learning pattern:
1. **Tokenization**: Sequences encoded with `<|endoftext|>DIFF:\n{diff}\n\nCOMMIT MESSAGE:\n{message}<|endoftext|>`
2. **Batching**: Custom `custom_collate_fn` pads variable-length sequences and creates shifted targets (next-token prediction)
3. **Training Step**: Forward pass with `fp16` mixed precision, `CrossEntropyLoss`, gradient scaling, and learning rate warmup
4. **Evaluation**: Compute validation loss and accuracy every epoch
5. **Checkpointing**: Save every 500 steps + best model + per-epoch snapshots

## Critical Patterns & Conventions

### Data Cleaning (`clean_text()`)
The `clean_text()` function is essential for data quality:
- Removes git metadata (`index`, `---`, `+++` lines, author attribution)
- Simplifies diff headers to `FILE: filename` format
- Strips URLs and issue references (reduce token bloat)
- Preserves actual code changes (the `@@` context lines and `+`/`-` markers)
- **Important**: Called on both diffs and messages before concatenation

### Custom Collate Function
Implements left-padding with special handling:
- Pads to maximum sequence length in batch (not fixed size)
- Input: tokens `[0:n-1]`, Target: shifted tokens `[1:n]` (next-token prediction)
- Masks padding tokens in targets with `-100` to ignore in loss calculation
- **Key line**: `targets[indices[1:]] = ignore_index` ensures only the first padding token is used

### Training State Management
The script resumes from checkpoints:
- `latest_checkpoint.pt` saved every 500 steps (preserves optimizer, scheduler, scaler state)
- Per-epoch snapshots saved to `data/gpt2_epoch{N}.pt`
- Best model saved to `data/best_model.pt` on validation improvement
- Early stopping after 2 epochs without improvement

### Model Configuration
```python
model_name = "gpt2"
batch_size = 4
lr = 5e-5
epochs = 5
max_sequence_length = 512
optimizer = AdamW(weight_decay=0.01)
scheduler = linear_schedule_with_warmup (10% warmup)
mixed_precision = fp16 (GradScaler)
```

## Data Files & Structure
```
data/
├── commitbench_train_10pct.csv          # Training data
├── commitbench_validation_10pct.csv     # Validation data
├── train_tokenized_10pct.pt             # Cached tokenized training (rebuilt if missing)
├── val_tokenized_10pct.pt               # Cached tokenized validation
├── gpt2_epoch{1-5}.pt                   # Epoch checkpoints
└── best_model.pt                        # Best validation model
```
Cache files are regenerated if deleted; raw CSVs are filtered for diffs < 1000 chars and non-"Fixes #" messages.

## Development Workflows

### Running Training
```bash
python train_llm.py
```
- Logs to `training_log.txt` with timestamps
- Generates sample predictions every 500 steps from validation set
- Resume on interrupt: automatic checkpoint detection in `start()` function

### Adding New Features
- **New metrics**: Add to `train_one_epoch()` progress bar updates and `log_text()` calls
- **New data columns**: Update CSV loading in `start()` and field references in `prepare_and_save_data()`
- **Model changes**: Modify `model_name` in `start()` (currently GPT-2), adjust config params like `pad_token_id`

## Integration Points

### External Dependencies
- `transformers`: AutoTokenizer, AutoModelForCausalLM (HuggingFace)
- `torch`: Model training, mixed precision (`torch.amp.GradScaler`)
- `pandas`: CSV loading and filtering
- `tqdm`: Progress bars

### Key Functions (Entry Points)
- `start()`: Main initialization and training launcher
- `train()`: Outer epoch loop with early stopping logic
- `train_one_epoch()`: Step-level training with checkpointing every 500 steps
- `evaluate()`: Validation loop
- `generate()`: Sample generation from validation diffs (diagnostic)

## Common Debugging
- **OOM errors**: Reduce `batch_size` or `max_sequence_length` in tokenization
- **Low accuracy**: Check `clean_text()` isn't removing important context; verify data format matches expected CSV columns
- **Resume failures**: Ensure `latest_checkpoint.pt` is compatible with code changes; delete to restart from epoch 0
- **Training hung**: Check CUDA availability with `torch.cuda.is_available()` in device assignment