import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import random

# Configuration
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "models/fine_tuned_nllb"
TRAIN_FILE = "data/train.json"
VAL_FILE = "data/val.json"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "fra_Latn"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['source']
        target_text = item['target']
        
        # Tokenize source and target
        # NLLB requires src_lang and tgt_lang to be set or passed.
        # We use text_target for the target text.
        
        model_inputs = self.tokenizer(
            source_text, 
            text_target=target_text,
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
            
        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": model_inputs["labels"].squeeze()
        }

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Add special tokens for inline annotations
    special_tokens_dict = {'additional_special_tokens': ['<dnt>', '</dnt>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens: {special_tokens_dict}")
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        # Important: We need to train the new embeddings.
        # LoRA freezes the base model, so we must explicitly tell it to train the embedding layers
        # or at least the new tokens. However, standard LoRA implementation might not support 
        # training specific tokens easily without 'modules_to_save'.
        # We will add 'embed_tokens' and 'lm_head' to modules_to_save to ensure new tokens are trained.
        modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # Load Data
    print("Loading data...")
    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)
    
    train_dataset = TranslationDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = TranslationDataset(val_data, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )
    
    # Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                val_loss += outputs.loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        model.train()
        
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    main()
