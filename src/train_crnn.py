import json
from pathlib import Path
from time import time

import torch
from torch import nn, optim

from src.dataset import get_dataloaders
from src.models import CNNLSTM


def get_device():
    if torch.backends.mps.is_available():
        print("Device : mps (Apple GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Device : cuda (GPU)")
        return torch.device("cuda")
    else:
        print("Device : cpu")
        return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    print(f"  -> nb batches train : {len(loader)}")

    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, y)

        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            print(f"    batch {batch_idx+1}/{len(loader)}")

    epoch_loss = running_loss / n_batches
    epoch_acc = running_acc / n_batches
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            acc = accuracy_from_logits(logits, y)

            running_loss += loss.item()
            running_acc += acc
            n_batches += 1

    epoch_loss = running_loss / n_batches
    epoch_acc = running_acc / n_batches
    return epoch_loss, epoch_acc


def main():
    # ========================
    # 0. Config de base
    # ========================
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    CHECKPOINT_DIR = ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

    # Hyperparamètres
    batch_size = 32
    num_workers = 0
    num_epochs = 10        # tu peux ajuster
    learning_rate = 1e-3
    n_mels = 128
    n_classes = 8

    device = get_device()

    # ========================
    # 1. DataLoaders
    # ========================
    loaders = get_dataloaders(
        data_dir=DATA_DIR,
        csv_name="features_mels.csv",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        train_subset_size=None,      # ou 1000 si tu veux aller plus vite
        augment_train=True,          # tu peux activer/désactiver SpecAugment ici
    )

    train_loader = loaders["train"]
    valid_loader = loaders["valid"]
    test_loader = loaders["test"]

    print(f"Train batches : {len(train_loader)}")
    print(f"Valid batches : {len(valid_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # ========================
    # 2. Modèle, loss, optim
    # ========================
    model = CNNLSTM(n_mels=n_mels, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ========================
    # 3. Boucle d'entraînement
    # ========================
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    best_valid_acc = 0.0
    best_model_path = CHECKPOINT_DIR / "crnn_best.pt"

    print("Début de l'entraînement (CRNN : CNN + LSTM)...")
    start_time = time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        t0 = time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion, device
        )

        epoch_time = time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"  Train - loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
            f"Valid - loss: {valid_loss:.4f}, acc: {valid_acc:.2f}%"
        )
        print(f"  Temps epoch : {epoch_time:.1f} s")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_valid_acc": best_valid_acc,
                    "n_mels": n_mels,
                    "n_classes": n_classes,
                },
                best_model_path,
            )
            print(f"  👉 Nouveau meilleur modèle sauvegardé ({best_valid_acc:.2f}%)")

        # sauvegarde de l'historique à chaque epoch
        history_path = CHECKPOINT_DIR / "crnn_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = time() - start_time
    print(f"\nEntraînement terminé en {total_time/60:.1f} minutes.")
    print(f"Meilleure validation accuracy (CRNN) : {best_valid_acc:.2f}%")

    print("\nÉvaluation sur le test set (CRNN)...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test (CRNN) - loss: {test_loss:.4f}, acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
