import torch
import torch.nn as nn


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


class CNNLSTM(nn.Module):
    """
    Modèle CRNN : CNN (sur temps-fréquence) + LSTM (sur le temps).

    - Le CNN extrait des features locales (temps-fréquence)
    - On moyenne sur la dimension fréquence
    - On envoie la séquence temporelle dans un LSTM
    - On utilise le dernier état comme représentation globale
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 8,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()

        c1, c2, c3 = cnn_channels

        # Blocs CNN (similaires au baseline)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # /2 sur freq et temps
            nn.Dropout(0.1),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
        )

        # On garde la dimension temps, on va juste réduire la dimension fréquence
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, T')

        self.bidirectional = bidirectional
        lstm_input_size = c3
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,      # input: (B, T', C)
            bidirectional=bidirectional,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor de shape (B, 1, 128, T)
        :return: logits (B, n_classes)
        """
        # CNN
        x = self.conv_block1(x)  # (B, c1, F1, T1)
        x = self.conv_block2(x)  # (B, c2, F2, T2)
        x = self.conv_block3(x)  # (B, c3, F3, T3)

        # Pooling sur la fréquence -> (B, c3, 1, T3)
        x = self.freq_pool(x)

        # On enlève la dimension fréquence (1) : (B, c3, T3)
        x = x.squeeze(2)

        # On permute pour avoir la séquence temporelle comme dimension principale : (B, T3, c3)
        x = x.transpose(1, 2)

        # LSTM
        # output : (B, T3, D * H), h_n : (num_layers * num_directions, B, H)
        output, (h_n, c_n) = self.lstm(x)

        # On prend le dernier état caché (t final)
        if self.bidirectional:
            # concat des deux directions
            last_hidden = torch.cat(
                (h_n[-2], h_n[-1]), dim=1
            )  # (B, 2 * H)
        else:
            last_hidden = h_n[-1]  # (B, H)

        # Classification
        logits = self.classifier(last_hidden)  # (B, n_classes)
        return logits


if __name__ == "__main__":
    # petit test
    B = 4
    n_mels = 128
    T = 1000

    x = torch.randn(B, 1, n_mels, T)

    print("=== Test BaselineCNN ===")
    baseline = BaselineCNN(n_mels=n_mels, n_classes=8)
    out_base = baseline(x)
    print("Baseline output shape :", out_base.shape)  # (4, 8)

    print("\n=== Test CNNLSTM ===")
    crnn = CNNLSTM(n_mels=n_mels, n_classes=8)
    out_crnn = crnn(x)
    print("CNNLSTM output shape :", out_crnn.shape)   # (4, 8)
