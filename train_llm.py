#from google.colab import drive
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
import datetime
import os
from tqdm.auto import tqdm
import math
import torch.nn as nn


def clean_text(text):
# 1. Αφαίρεση της γραμμής index (είτε έχει πραγματικό hash είτε το λεκτικό <HASH>)
    # Χρησιμοποιούμε το .* για να πιάσουμε τα πάντα μέχρι την αλλαγή γραμμής
    text = re.sub(r'^index .*\n', '', text, flags=re.MULTILINE)
    
    # 2. Απλοποίηση του 'diff --git' σε 'FILE:'
    # Κρατάμε μόνο το όνομα του αρχείου. Είναι το πιο σημαντικό context!
    text = re.sub(r'^diff --git a/(.*) b/(.*)\n', r'FILE: \1\n', text, flags=re.MULTILINE)
    
    # 3. Αφαίρεση των γραμμών --- a/ και +++ b/ 
    # Αφού έχουμε το FILE:, αυτές οι γραμμές είναι 100% περιττές.
    text = re.sub(r'^--- a/.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\+\+\+ b/.*\n', '', text, flags=re.MULTILINE)


    # 5. Καθαρισμός κενών γραμμών που προέκυψαν από τις αφαιρέσεις
    text = re.sub(r'\n\s*\n', '\n', text)

    # 1. Αφαίρεση metadata (Signed-off-by, Co-authored-by, κλπ)
    # Αυτά καταστρέφουν το training γιατί το μοντέλο μαθαίνει ονόματα αντί για κώδικα
    text = re.sub(r'^(Signed-off-by|Co-authored-by|Reported-by|Reviewed-by|Cc):.*$', '', text, flags=re.MULTILINE)
    
    # 2. Αφαίρεση links προς Issues ή Pull Requests (π.χ. https://github.com...)
    # Τα URL είναι τεράστια σε tokens και δεν προσφέρουν νόημα στο GPT-2
    text = re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)
    
    # 3. Αφαίρεση των placeholders <HASH> ή <I> αν υπάρχουν μόνα τους
    text = text.replace('<HASH>', '').replace('<I>', '')
    
    # 4. Καθαρισμός πολλαπλών κενών και αλλαγών γραμμής
    #text = re.sub(r'\s+', ' ', text)

    return text.strip()


class CodeDiffMessageDataset(Dataset):
    def __init__(self, data, tokenizer,max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 512


    def __getitem__(self, index):
        entry = self.data.iloc[index]
        
        # Clean the components
        clean_diff = clean_text(entry['diff'])
        clean_msg = clean_text(entry['message'])

        # Structure: <|endoftext|> DIFF: ... MESSAGE: ... <|endoftext|>
        prompt = f"<|endoftext|>DIFF:\n{clean_diff}\n\nCOMMIT MESSAGE:\n"
        full_text = f"{prompt}{clean_msg}<|endoftext|>"

        encoded_full = self.tokenizer.encode(full_text, max_length=self.max_length, truncation=True)
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_length, truncation=True)
        
        return  encoded_full


    def __len__(self):
        return int(len(self.data))


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index


        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return inputs_tensor, targets_tensor

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1) #(batch_size, sequence_length, vocab_size)
    correct_predictions = (predictions == labels).float() # (batch_size, sequence_length)
    return correct_predictions.mean().item()

def get_random_validation_diff(val_df):
    # Διαλέγουμε μια τυχαία γραμμή από το validation dataframe
    random_row = val_df.sample(n=1).iloc[0]
    return random_row['diff'], random_row['message']

def generate(model, tokenizer, device,val_df):
    model.eval()
    val_diff, actual_message = get_random_validation_diff(val_df)
    val_diff = clean_text(val_diff)
    actual_message = clean_text(actual_message)
    prompt = f"<|endoftext|>DIFF:\n{val_diff}\n\nCOMMIT MESSAGE:\n"
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=512, 
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=40,
            do_sample=True,      
            top_p=0.9,
            repetition_penalty=1.2, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Αποκωδικοποίηση μόνο της απάντησης
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_message = full_text.split("COMMIT MESSAGE:")[-1].strip()

    log_text("\n" + "="*50)
    log_text(f"TEST ON RANDOM VAL DIFF:")
    log_text(f"DIFF:\n{val_diff}")
    log_text(f"\nACTUAL MESSAGE: {actual_message}")
    log_text(f"GENERATED MESSAGE: {generated_message}")
    log_text("="*50 + "\n")
    model.train()

# --- 2. Training Step ---
def train_one_epoch(model, dataloader, optimizer, device, progress_bar, tokenizer,scheduler,val_df,start_epoch,epoch,global_step):
    model.train()
    total_loss = 0
    total_acc = 0
    

    for i, (x, y) in enumerate(dataloader):
        if (epoch == start_epoch) and (i < (global_step % len(dataloader))):
            progress_bar.update(1)
            continue

        
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x).logits
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        acc = calculate_accuracy(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc

        progress_bar.update(1)
        progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}","train_acc": f"{acc:.4f}","lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        global_step += 1
        if global_step > 0 and global_step % 500 == 0:
            generate(model, tokenizer, device,val_df)
            checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
            torch.save(checkpoint, "latest_checkpoint.pt")
            log_text(f"Checkpoint saved at step {global_step}")


    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss,avg_acc,global_step

# --- 3. Evaluation Step ---
def evaluate(model, dataloader, device,progress_bar):
    model.eval()
    total_loss = 0
    total_acc=0
    with torch.no_grad():

        for i,(x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x).logits
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()
            val_acc = calculate_accuracy(logits, y)
            total_acc  += val_acc
            progress_bar.update(1)
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_accuracy": f"{val_acc:.4f}"})


    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss,avg_acc


# --- 4. Main Loop ---
def train(model, train_loader, val_loader, optimizer,scheduler, device, tokenizer,epochs,val_df):
    best_val_loss = float("inf")
    patience = 2
    bad_epochs = 0
    checkpoint_path = "latest_checkpoint.pt"
    start_epoch = 0
    global_step = 0
    main_path = "data/"
    # ΕΛΕΓΧΟΣ ΓΙΑ RESUME
    if os.path.exists(checkpoint_path):
        log_text(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        log_text(f"Resuming from Epoch {start_epoch}, Global Step {global_step}")

    for epoch in range(start_epoch,epochs):

        progress_bar = tqdm(enumerate(train_loader), 
                            total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{epochs}",
                            dynamic_ncols=True,
                            mininterval=2.0,
                            maxinterval=3.0 
                            )

        # Train
        train_loss,train_acc,global_step = train_one_epoch(model, 
                                                          train_loader,    
                                                          optimizer, 
                                                          device, 
                                                          progress_bar, 
                                                          tokenizer,
                                                          scheduler,
                                                          val_df,
                                                          start_epoch,
                                                          epoch,
                                                          global_step
                                                          )

        # Evaluate
        val_loss,val_acc = evaluate(model, 
                                    val_loader, 
                                    device,
                                    progress_bar)
        try:
            ppl = math.exp(val_loss)
        except OverflowError:
            ppl = float('inf')

        progress_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_accuracy": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_accuracy": f"{val_acc:.4f}",
            "PPL": f"{ppl:.2f}"
        })
        progress_bar.refresh()
        progress_bar.close() # Close bar to log_text new line

        # Logging
        log_text(f"Summary Epoch {epoch+1}: Train Loss: {train_loss:.4f}| Train Accuracy: {train_acc:.4f}\
        | Val Loss: {val_loss:.4f}| Val Accuracy: {val_acc:.4f})| PPL: {ppl:.4f}")

        # Save checkpoints
        torch.save(model.state_dict(), f"{main_path}/gpt2_epoch{epoch+1}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), f"{main_path}/best_model.pt")
            log_text(">>> New best model saved!")
        else:
            bad_epochs += 1
            log_text(f">>> No improvement (bad epochs: {bad_epochs})")

        if bad_epochs >= patience:
            log_text("EARLY STOPPING TRIGGERED.")
            break


def log_text(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    

    with open("training_log.txt", "a", encoding="utf-8") as f:
        f.write(formatted_message)
        f.flush() 
    
    print(formatted_message) 


def start():
    main_path = "data/"
    os.makedirs(main_path, exist_ok=True)
    train_df = pd.read_csv(main_path + "commitbench_train_10pct.csv")
    val_df = pd.read_csv(main_path + "commitbench_validation_10pct.csv")
    log_text(f"Length Train:{len(train_df)}")
    log_text(f"Length Val:{len(val_df)}")

    train_df = train_df[train_df["diff"].str.len() < 1000].reset_index(drop=True)
    val_df = val_df[val_df["diff"].str.len() < 1000].reset_index(drop=True)
    train_df = train_df[~train_df['message'].str.contains('^Fixes #', na=False)]

    log_text(f"New Length Train: {len(train_df)}")
    log_text(f"New Length Val: {len(val_df)}")


    num_workers = 0
    batch_size = 4

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(seed)
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = CodeDiffMessageDataset(train_df, tokenizer)
    val_dataset = CodeDiffMessageDataset(val_df, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle= False,
        drop_last=True,
        num_workers=num_workers
        pin_memory=True
    )



    
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    model.config.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr = 5e-5
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    train(model, train_loader, val_loader, optimizer, scheduler ,device, tokenizer,epochs,val_df)

if __name__ == "__main__":
    start()