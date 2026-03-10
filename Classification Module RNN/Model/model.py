import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for classifying underwater audio mel spectrograms.

    Input shape:  (batch, 1, N_MELS, time_steps)
                   batch = number of 3s audio chunks per batch
                   1     = single channel (grayscale image)
                   N_MELS      = 64  (frequency axis)
                   time_steps  = 3*32000 // 512 + 1 = 188

    Output shape: (batch, num_classes)

    Architecture:
        Block 1: Conv → ReLU → MaxPool → Dropout   (finds basic patterns)
        Block 2: Conv → ReLU → MaxPool → Dropout   (finds combinations)
        Block 3: Conv → ReLU → MaxPool → Dropout   (finds class-level patterns)
        Flatten
        FC → ReLU → Dropout
        FC → ReLU → Dropout
        Output (num_classes)
    """

    def __init__(self, num_classes, n_mels=64, time_steps=188):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # After 3 x MaxPool2d(2), each dimension is divided by 8
        flat_size = 64 * (n_mels // 8) * (time_steps // 8)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
