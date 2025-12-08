from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ==== Fonctions d'augmentation (SpecAugment light) ====

def time_mask(spec: np.ndarray, max_width: int = 50) -> np.ndarray:
    """
    Masque aléatoirement une bande temporelle sur le spectrogramme.
    spec: (n_mels, T)
    """
    n_mels, T = spec.shape
    t = np.random.randint(0, max_width + 1)
    if t == 0 or t >= T:
        return spec
    t0 = np.random.randint(0, T - t + 1)
    spec_aug = spec.copy()
    spec_aug[:, t0:t0 + t] = 0.0
    return spec_aug


def freq_mask(spec: np.ndarray, max_width: int = 15) -> np.ndarray:
    """
    Masque aléatoirement une bande de fréquences.
    spec: (n_mels, T)
    """
    n_mels, T = spec.shape
    f = np.random.randint(0, max_width + 1)
    if f == 0 or f >= n_mels:
        return spec
    f0 = np.random.randint(0, n_mels - f + 1)
    spec_aug = spec.copy()
    spec_aug[f0:f0 + f, :] = 0.0
    return spec_aug


def spec_augment(
    spec: np.ndarray,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    max_time_width: int = 50,
    max_freq_width: int = 15,
) -> np.ndarray:
    """
    Applique plusieurs masques temps/fréquence (SpecAugment simplifié).
    """
    out = spec
    for _ in range(num_time_masks):
        out = time_mask(out, max_width=max_time_width)
    for _ in range(num_freq_masks):
        out = freq_mask(out, max_width=max_freq_width)
    return out


class MusicDataset(Dataset):
    """
    Dataset PyTorch pour les melspectrogrammes FMA-small.

    Chaque item retourne :
      - X : Tensor float32 de shape (1, n_mels, T)
      - y : Tensor long (label de genre)
    """

    def __init__(self, csv_path: str | Path, split: str = "train", augment: bool = False):
        super().__init__()
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")

        df = pd.read_csv(csv_path)

        if "split" not in df.columns:
            raise ValueError("Le CSV doit contenir une colonne 'split'.")
        if "mel_path" not in df.columns:
            raise ValueError("Le CSV doit contenir une colonne 'mel_path'.")
        if "genre_id" not in df.columns:
            raise ValueError("Le CSV doit contenir une colonne 'genre_id'.")

        if split not in ("train", "valid", "test"):
            raise ValueError("split doit être 'train', 'valid' ou 'test'.")

        # mapping FMA -> splits simples
        mapping = {
            "training": "train",
            "validation": "valid",
            "test": "test",
        }
        df["split_simple"] = df["split"].map(lambda s: mapping.get(s, s))

        self.df = df[df["split_simple"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"Aucune donnée pour le split '{split}' "
                f"(valeurs uniques trouvées: {df['split_simple'].unique()})"
            )

        self.split = split
        # on n'active réellement l'augmentation que pour le split train
        self.augment = augment and (split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        mel_path = Path(row["mel_path"])
        if not mel_path.exists():
            raise FileNotFoundError(f"Mel file introuvable : {mel_path}")

        mel = np.load(mel_path)  # (n_mels, T)

        # 🔥 augmentation uniquement en train si activée
        if self.augment:
            mel = spec_augment(mel)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)

        label = int(row["genre_id"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor


def get_dataloaders(
    data_dir: str | Path = "data",
    csv_name: str = "features_mels.csv",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_subset_size: int | None = None,
    augment_train: bool = False,
):
    """
    Crée des DataLoader pour train / valid / test à partir du CSV.

    :param data_dir: dossier où se trouve le CSV (par défaut "data/")
    :param csv_name: nom du fichier CSV (par défaut "features_mels.csv")
    :param batch_size: taille de batch
    :param num_workers: nb de workers pour le DataLoader (0 recommandé sur Mac/MPS)
    :param pin_memory: True si tu utilises un GPU CUDA
    :param train_subset_size: si non-None, nombre d'exemples de train à garder (tirage aléatoire)
    :param augment_train: si True, applique SpecAugment sur les mels du train
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / csv_name

    train_dataset = MusicDataset(csv_path, split="train", augment=augment_train)
    valid_dataset = MusicDataset(csv_path, split="valid", augment=False)
    test_dataset = MusicDataset(csv_path, split="test", augment=False)

    # Sous-échantillonnage du train si demandé
    if train_subset_size is not None and train_subset_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_subset_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    # petit test rapide
    loaders = get_dataloaders(
        data_dir="data",
        csv_name="features_mels.csv",
        batch_size=8,
        num_workers=0,
        train_subset_size=200,
        augment_train=True,
    )
    train_loader = loaders["train"]
    print("Nb de batches train :", len(train_loader))
    X, y = next(iter(train_loader))
    print("Batch X shape :", X.shape)
    print("Batch y shape :", y.shape)
