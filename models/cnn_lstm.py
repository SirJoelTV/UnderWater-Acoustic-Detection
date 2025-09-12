import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(attention_output)
        return output

class UnderwaterCNNLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # CNN for spatial feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for filters in config['model']['cnn_lstm']['cnn_filters']:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(config['model']['cnn_lstm']['dropout'])
            ))
            in_channels = filters
        
        # Calculate feature size after CNN
        self.feature_size = self._get_conv_output_size()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=config['model']['cnn_lstm']['lstm_units'],
            num_layers=2,
            batch_first=True,
            dropout=config['model']['cnn_lstm']['dropout'],
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            d_model=config['model']['cnn_lstm']['lstm_units'] * 2,  # bidirectional
            n_heads=8
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['cnn_lstm']['lstm_units'] * 2, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['cnn_lstm']['dropout']),
            nn.Linear(512, config['model']['cnn_lstm']['num_classes'])
        )
        
    def _get_conv_output_size(self):
        # Dummy forward pass to calculate size
        x = torch.randn(1, 1, 128, 125)  # (batch, channels, mel_bins, time_steps)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x.view(x.size(0), x.size(1), -1).size(-1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        # Input: (batch, mel_bins, time_steps)
        x = x.unsqueeze(1)  # Add channel dimension
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM: (batch, time_steps, features)
        x = x.view(batch_size, x.size(1), -1)  # Flatten spatial dimensions
        x = x.transpose(1, 2)  # (batch, time_steps, features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_features = self.attention(lstm_out)
        
        # Global average pooling over time dimension
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output, pooled_features  # Return features for anomaly detection

class MultiModalCNNLSTM(nn.Module):
    """Multi-modal version that handles mel-spec, MFCC, and LOFAR features"""
    def __init__(self, config):
        super().__init__()
        
        # Separate CNN branches for different features
        self.mel_branch = UnderwaterCNNLSTM(config)
        self.mfcc_branch = UnderwaterCNNLSTM(config)
        self.lofar_branch = UnderwaterCNNLSTM(config)
        
        # Fusion layer
        feature_dim = config['model']['cnn_lstm']['lstm_units'] * 2 * 3  # 3 branches
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['cnn_lstm']['dropout']),
            nn.Linear(512, config['model']['cnn_lstm']['num_classes'])
        )
        
    def forward(self, mel_spec, mfcc, lofar):
        # Process each modality
        mel_out, mel_features = self.mel_branch(mel_spec)
        mfcc_out, mfcc_features = self.mfcc_branch(mfcc)
        lofar_out, lofar_features = self.lofar_branch(lofar)
        
        # Fuse features
        fused_features = torch.cat([mel_features, mfcc_features, lofar_features], dim=1)
        final_output = self.fusion_layer(fused_features)
        
        return final_output, fused_features