import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from preprocess import midi_tokenize, preprocess_midi, preprocess_pluggnb
from decoder import melody_transformer, melody_transformer_pretrained_small
from transformers import Trainer, TrainingArguments

import matplotlib.pyplot as plt


# =========================== #
# two-part fine tune training #
# =========================== #

# flash attention?

def train(model, dataloader, criterion, optimizer, scaler, epoch, grad_accumulation_steps):
    model.train()
    total_loss = 0

    for i, batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} Training."):
        inputs = batch.to(device) # ensure batch is sent to gpu to parallelize
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model.forward(inputs, labels=inputs) # predict using input sentence and (shifted) input sentence as labels
            loss = criterion.forward(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1)) # compute cross-entropy loss
        
        scaler.scale(loss).backward()

        optimizer.zero_grad()

        # if (i+1) % grad_accumulation_steps == 0: # optimize gpu memory (occasionally accumulate gradients)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() # accumulate loss to get average across batches
        print(f"Train Loss: {loss.item()}")
    
    return total_loss / len(dataloader) # total average loss across the dataloader batches.


def eval(model, dataloader, criterion, epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad(): # validation 
        for i, batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} Evaluating."):
            inputs = batch.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model.forward(inputs, labels=inputs)
                loss = criterion.forward(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
            
            total_loss += loss.item()
            print(f"Val Loss: {loss.item()}")
        
    return total_loss / len(dataloader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# phase 1 of fine-tune: overhaul language sequence relationships with musical note sequence relationships (~4 layers) #
print("---Training Phase 1 Commencing...---")

dataloader = preprocess_midi()
model = melody_transformer_pretrained_small()

for param in model.parameters(): # freeze all layers
    param.requires_grad = False

finetune_layers = [] # store the layers we want to train

for i in range (-4, 0): # unfreeze only the last 4 layers
    for param in model.transformer.h[i].parameters():
        param.requires_grad = True
        finetune_layers.append(param)

# hyperparameters
best_val_loss = float("inf")
best_model = model.state_dict() # track best performing

num_epochs = 10
learning_rate = 1e-4
epsilon = 1e-9
grad_accumulation_steps = 2

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(finetune_layers, lr=learning_rate, eps=epsilon)

# we want to use mixed-precision training to take full advantage of our nvidia tensor cores
scaler = torch.amp.GradScaler() # implements mixed precision (fp16)

p1_losses = [] # track best instance losses

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_loss = train(model, dataloader, criterion, optimizer, scaler, num_epochs, grad_accumulation_steps) # accumulate gradients
    val_loss = eval(model, dataloader, criterion, epoch) # no gradients

    print(f"Total Average rain Loss: {train_loss:.4f}")
    print(f"Total Average Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss: # we grade model by validation and save best instance
        best_val_loss = val_loss
        best_model = model.state_dict()

fig = plt.figure(figsize=(20,10))
ax = fig.subplots(1)
ax.plot(p1_losses)

torch.save(best_model, Path("models", "melody_transformer.pth"))


# phase 2 of fine-tune: fine tune new melody-aware transformer on pluggnb #
print("---Training Phase 2 Commencing...---")

dataloader = preprocess_pluggnb()
model.load_state_dict(torch.load(Path("models", "melody_transformer.pth")), map_location = device)
model.to(device)

# hyperparameters
num_epochs = 5
learning_rate = 1e-5

for param in model.parameters(): # freeze all layers
    param.requires_grad = False

for param in model.lm_head.parameters(): # unfreeze final layer
    param.requires_grad = True

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_loss = train(model, dataloader, criterion, optimizer, scaler, num_epochs, grad_accumulation_steps) # accumulate gradients
    val_loss = eval(model, dataloader, criterion, epoch) # no grad

    print(f"Total Average Train Loss: {train_loss:.4f}")
    print(f"Total Average Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss: # save best instance
        best_val_loss = val_loss
        best_model = model.state_dict()

torch.save(model.state_dict(), Path("models", "pluggnb_transformer.pth"))
