import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 128, time_steps)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate the flattened size
        self.flatten_size = self._get_flatten_size()
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, config['model']['autoencoder']['latent_dim']),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(config['model']['autoencoder']['latent_dim'], self.flatten_size),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
    def _get_flatten_size(self):
        # Calculate size after encoder
        x = torch.randn(1, 1, 128, 125)
        x = self.encoder(x)
        return x.numel()
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        latent = self.bottleneck(encoded)
        
        # Decode
        decoded_linear = self.decoder_linear(latent)
        
        # Reshape for decoder
        batch_size = x.size(0)
        decoded_reshaped = decoded_linear.view(batch_size, 256, 
                                             encoded.size(2), encoded.size(3))
        
        # Final decode
        reconstructed = self.decoder(decoded_reshaped)
        
        return reconstructed, latent
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        reconstructed, _ = self.forward(x)
        
        # Mean squared error per sample
        mse = F.mse_loss(reconstructed, x.unsqueeze(1), reduction='none')
        error = mse.view(mse.size(0), -1).mean(dim=1)
        
        return error

class AnomalyDetector:
    def __init__(self, autoencoder, threshold_percentile=95):
        self.autoencoder = autoencoder
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        
    def fit_threshold(self, normal_data_loader):
        """Fit anomaly threshold on normal data"""
        self.autoencoder.eval()
        errors = []
        
        with torch.no_grad():
            for batch in normal_data_loader:
                # Use mel_spec as input
                mel_spec = batch['mel_spec']
                error = self.autoencoder.get_reconstruction_error(mel_spec)
                errors.extend(error.cpu().numpy())
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")
        
    def detect_anomaly(self, x):
        """Detect anomalies in new data"""
        self.autoencoder.eval()
        with torch.no_grad():
            error = self.autoencoder.get_reconstruction_error(x)
            
        if self.threshold is None:
            raise ValueError("Threshold not fitted. Call fit_threshold first.")
            
        is_anomaly = error > self.threshold
        confidence = (error / self.threshold).clamp(0, 2)  # Scale to 0-2
        
        return is_anomaly, confidence, error