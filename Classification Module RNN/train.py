import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json

import config
from model import EnhancedBiLSTM
from preprocessing import UnderwaterDataset

def train():
    device = config.get_device()
    
    # 1. Prepare Data
    full_ds = UnderwaterDataset(config.DATA_DIR, training=True)
    
    # Save classes for inference later
    with open(config.CLASSES_PATH, 'w') as f:
        json.dump(list(full_ds.encoder.classes_), f)
    print(f"Saved class labels to {config.CLASSES_PATH}")

    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    model = EnhancedBiLSTM(num_classes=len(full_ds.encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 3. Training Loop
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.1f}% | Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print("--> Model Saved")

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total, loss_sum / len(loader)

if __name__ == "__main__":
    train()