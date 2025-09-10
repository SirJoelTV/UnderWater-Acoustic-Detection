import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import os

from data.preprocessing import create_dataloaders
from models.cnn_lstm import MultiModalCNNLSTM
from models.autoencoder import ConvolutionalAutoEncoder, AnomalyDetector
from utils.visualization import plot_training_curves, plot_confusion_matrix

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.classifier = MultiModalCNNLSTM(config).to(self.device)
        self.autoencoder = ConvolutionalAutoEncoder(config).to(self.device)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Optimizers
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.autoencoder_optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate schedulers
        self.classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.classifier_optimizer, patience=5, factor=0.5
        )
        
        self.autoencoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.autoencoder_optimizer, patience=5, factor=0.5
        )
        
        # Tensorboard
        self.writer = SummaryWriter('runs/underwater_acoustic_detection')
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.classifier.train()
        self.autoencoder.train()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            mel_spec = batch['mel_spec'].to(self.device)
            mfcc = batch['mfcc'].to(self.device)
            lofar = batch['lofar'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # Classification training
            self.classifier_optimizer.zero_grad()
            
            class_output, features = self.classifier(mel_spec, mfcc, lofar)
            class_loss = self.classification_loss(class_output, labels)
            
            class_loss.backward()
            self.classifier_optimizer.step()
            
            # Autoencoder training (on mel spectrograms)
            self.autoencoder_optimizer.zero_grad()
            
            reconstructed, latent = self.autoencoder(mel_spec)
            recon_loss = self.reconstruction_loss(reconstructed, mel_spec.unsqueeze(1))
            
            recon_loss.backward()
            self.autoencoder_optimizer.step()
            
            # Total loss for logging
            batch_loss = class_loss.item() + recon_loss.item()
            total_loss += batch_loss
            
            # Update progress bar
            pbar.set_postfix({
                'Class Loss': f'{class_loss.item():.4f}',
                'Recon Loss': f'{recon_loss.item():.4f}',
                'Total Loss': f'{batch_loss:.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Classification', class_loss.item(), global_step)
            self.writer.add_scalar('Loss/Reconstruction', recon_loss.item(), global_step)
            
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.classifier.eval()
        self.autoencoder.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spec'].to(self.device)
                mfcc = batch['mfcc'].to(self.device)
                lofar = batch['lofar'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Classification
                class_output, _ = self.classifier(mel_spec, mfcc, lofar)
                class_loss = self.classification_loss(class_output, labels)
                
                # Reconstruction
                reconstructed, _ = self.autoencoder(mel_spec)
                recon_loss = self.reconstruction_loss(reconstructed, mel_spec.unsqueeze(1))
                
                total_loss += (class_loss.item() + recon_loss.item())
                
                # Accuracy calculation
                _, predicted = torch.max(class_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        # Log to tensorboard
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        for epoch in range(self.config['training']['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}')
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            self.classifier_scheduler.step(val_loss)
            self.autoencoder_scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best models
                torch.save(self.classifier.state_dict(), 'best_classifier.pth')
                torch.save(self.autoencoder.state_dict(), 'best_autoencoder.pth')
                print(f'New best model saved with validation loss: {val_loss:.4f}')
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Plot training curves
        plot_training_curves(self.train_losses, self.val_losses, self.val_accuracies)
        
        self.writer.close()
        print('Training completed!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, label_encoder = create_dataloaders(
        args.data_dir, config
    )
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Train model
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()