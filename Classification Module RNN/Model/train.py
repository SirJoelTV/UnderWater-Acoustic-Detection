import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

import config
from model import EnhancedBiLSTM
from preprocessing import create_stratified_split, save_global_statistics

def train():
    device = config.get_device()
    
    # 1. Prepare Data with Stratified Split
    print("Creating stratified train/val split...")
    train_ds, val_ds, classes = create_stratified_split(
        config.DATA_DIR, 
        test_size=0.2,
        categories=['ships', 'marine_life']
    )
    
    # Save classes for inference later
    with open(config.CLASSES_PATH, 'w') as f:
        json.dump(list(classes), f)
    print(f"Saved class labels to {config.CLASSES_PATH}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Setup Model with Class Weights to Handle Category Imbalance
    model = EnhancedBiLSTM(num_classes=len(classes)).to(device)
    
    # Compute class weights (inverse of class frequency)
    # This balances the 10 marine_life vs 4 ship categories
    train_targets = torch.tensor(train_ds.encoded_labels, dtype=torch.long)
    unique, counts = torch.unique(train_targets, return_counts=True)
    class_weights = torch.zeros(len(classes))
    total_samples = counts.sum().float()
    for idx, count in zip(unique, counts):
        class_weights[idx] = total_samples / (len(classes) * count.float())
    class_weights = class_weights.to(device)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # Use CosineAnnealingLR with warmup for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.EPOCHS, 
        eta_min=1e-5
    )

    # 3. Training Loop
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        # Warmup learning rate for first few epochs
        if epoch < config.WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * (epoch + 1) / config.WARMUP_EPOCHS
        
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        # Step scheduler after warmup
        if epoch >= config.WARMUP_EPOCHS:
            scheduler.step()
        
        # Validation
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{config.EPOCHS}: Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.4f}")

        # Save on best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  --> Model Saved (Best Val Acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter % 3 == 0:
                print(f"  --> No improvement for {patience_counter} epochs")
            
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered. Best Val Acc: {best_acc:.1f}%")
            break
    
    # Save global statistics for inference
    save_global_statistics(config.GLOBAL_STATS_PATH)
    print(f"\nTraining completed. Best validation accuracy: {best_acc:.1f}%")

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