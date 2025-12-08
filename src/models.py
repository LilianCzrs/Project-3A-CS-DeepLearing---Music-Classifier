import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN pour classifier des melspectrogrammes FMA-small.

    Entrée attendue : (batch_size, 1, n_mels=128, T)
    Sortie : logits de taille (batch_size, n_classes)
    """

    def __init__(self, n_mels: int = 128, n_classes: int = 8):
        super().__init__()

        # Bloc 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # /2 sur freq et temps
            nn.Dropout(0.1),
        )

        # Bloc 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        # Bloc 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
        )

        # On réduit tout à (channels, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Petit MLP de sortie
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (batch, 128, 1, 1) -> (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor de shape (B, 1, 128, T)
        :return: logits (B, n_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Petit test rapide
    batch_size = 4
    n_mels = 128
    T = 1000  # peu importe, pour tester

    model = BaselineCNN(n_mels=n_mels, n_classes=8)
    dummy_input = torch.randn(batch_size, 1, n_mels, T)
    out = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", out.shape)  # (4, 8)
