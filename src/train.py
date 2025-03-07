import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from preprocess import midi_tokenize, preprocess_midi, preprocess_pluggnb
from decoder import melody_transformer, melody_transformer_pretrained_small
from transformers import Trainer, TrainingArguments


# =========================== #
# two-part fine tune training #
# =========================== #

# flash attention?

def train(model, dataloader, criterion, optimizer, scaler, epoch, grad_accum_steps):
    model.train()
    total_loss = 0

    for i, batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
        inputs = batch.to(device) # ensure batch is sent to gpu to parallelize

        with torch.amp.autocast('cuda'):
            outputs = model(inputs, labels=inputs) # predict
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1), inputs.view(-1))) # compute loss
        
        scaler.scale(loss).backward()

        optimizer.zero_grad()

        if (i+1) % grad_accum_steps == 0: # optimize gpu memory (occasionally accumulate gradients)
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item() # accumulate loss to get average across batches
        print(f"Train Loss: {loss.item()}")
    
    return total_loss / len(dataloader)


def eval():
    model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

midi_tokenize()

# phase 1 of fine-tune: overhaul language sequence relationships with musical note sequence relationships (~4 layers) #
dataloader = preprocess_midi()

model = melody_transformer_pretrained_small()

for param in model.parameters(): # freeze all layers
    param.requires_grad = False

finetune_layers = [] # store the layers we want to train

for i in range (-4, 0): # unfreeze only the last 4 layers
    for param in model.transformer.h[i].parameters():
        param.requires_grad = True
        finetune_layers.append(param)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(finetune_layers, lr=1e-4, eps=1e-8)

# we want to use mixed-precision training to take full advantage of our nvidia tensor cores
scaler = torch.amp.GradScaler() # implements mixed precision (fp16)

# hyperparameters
num_epochs = 10
best_val_loss = float("inf")
best_model = model.state_dict() # track best performing

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_loss = train() # accumulate gradients
    val_loss = eval() # no gradients

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss: # we grade model by validation and save best instance
        best_val_loss = val_loss
        best_model = model.state_dict()

torch.save(best_model, Path("models", "melody_transformer.pth"))


# phase 2 of fine-tune: fine tune new melody-aware transformer on pluggnb #
dataloader = preprocess_pluggnb()

model = melody_transformer_pretrained_small()

model.load_state_dict(torch.load(Path("models", "melody_transformer.pth")), map_location = device)

model.to(device)

for param in model.parameters(): # freeze all layers
    param.requires_grad = False

for param in model.lm_head.parameters(): # unfreeze final layer
    param.requires_grad = True



torch.save(model.state_dict(), Path("models", "pluggnb_transformer.pth"))