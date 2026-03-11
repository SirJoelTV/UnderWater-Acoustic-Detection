import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import CNN_config as config
from CNN_model import SimpleCNN
from CNN_preprocessing import get_train_val_test_datasets


def get_time_steps():
    return (config.SAMPLE_RATE * config.CHUNK_DURATION) // config.HOP_LENGTH + 1


def train():
    print(f"Using device: {config.DEVICE}\n")

    # --- 1. Load data ---
    train_ds, val_ds, _, encoder = get_train_val_test_datasets(config.DATA_DIR)

    classes = list(encoder.classes_)
    with open(config.CLASSES_PATH, 'w') as f:
        json.dump(classes, f)
    print(f"\nSaved {len(classes)} classes.")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # --- 2. Build model ---
    time_steps = get_time_steps()
    model      = SimpleCNN(num_classes=len(classes),
                           n_mels=config.N_MELS,
                           time_steps=time_steps).to(config.DEVICE)
    print(f"Model ready. Input: {config.N_MELS} mels x {time_steps} time steps\n")

    # No class weights needed — classes balanced in preprocessing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=1e-4)

    # --- 3. Training loop ---
    best_val_acc               = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.EPOCHS):

        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss    = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total   += y.size(0)

        train_acc = 100 * train_correct / train_total

        # Validate
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1:>3}/{config.EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc               = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.CNN_MODEL_PATH)
            print(f"  --> Saved best model (Val Acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE:
                print(f"\nEarly stopping! No improvement for {config.PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y       = X.to(config.DEVICE), y.to(config.DEVICE)
            out        = model(X)
            total_loss += criterion(out, y).item()
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)
    return 100 * correct / total, total_loss / len(loader)


if __name__ == "__main__":
    train()