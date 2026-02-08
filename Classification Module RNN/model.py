import torch
import torch.nn as nn

class EnhancedBiLSTM(nn.Module):
    def __init__(self, num_classes, input_size=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True  # Reduced from 0.5
        )
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Add batch norm
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Add batch norm
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)  # Reduced from 0.6
        self.relu = nn.ReLU()

    def attention(self, x):
        # x shape: [batch, seq_len, hidden_dim*2]
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.bn1(self.fc1(x))
        x = self.dropout(self.relu(x))
        x = self.bn2(self.fc2(x))
        x = self.dropout(self.relu(x))
        return self.fc3(x)