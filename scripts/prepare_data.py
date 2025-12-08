"""
Script de préparation des données FMA-small :
- lecture des métadonnées (tracks.csv)
- sélection du subset "small"
- création des chemins audio
- extraction des melspectrogrammes normalisés
- sauvegarde en .npy
- création d'un CSV récapitulatif pour l'entraînement

Usage (depuis la racine du projet) :
    python scripts/prepare_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# =========================
# 1. Chemins & constantes
# =========================

# Dossier racine du projet (dépend de l'endroit où tu mets le script)
# Ici : scripts/prepare_data.py  -> racine = parent du parent
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "fma_small"       # dossier de fma_small.zip extrait
META_DIR = DATA_DIR / "fma_metadata"     # dossier qui contient tracks.csv, genres.csv, ...

MELS_DIR = DATA_DIR / "mels"
MELS_DIR.mkdir(exist_ok=True, parents=True)

tracks_path = META_DIR / "tracks.csv"

# Paramètres audio / mel
SR = 22050         # sample rate
N_MELS = 128       # imposé par l'énoncé
N_FFT = 2048
HOP_LENGTH = 512

# On fixe une durée cible (en secondes) pour avoir la même taille temporelle
TARGET_DURATION = 30.0  # 30 secondes par exemple
MAX_FRAMES = int(np.ceil(TARGET_DURATION * SR / HOP_LENGTH))


# =========================
# 2. Chargement des métadonnées
# =========================

def load_metadata(tracks_csv: Path) -> pd.DataFrame:
    """
    Lit tracks.csv (MultiIndex) et renvoie un DataFrame simplifié
    pour le subset "small" avec les colonnes utiles.
    """
    print(f"Lecture de {tracks_csv} ...")
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # On garde uniquement le subset "small"
    subset = tracks[tracks[('set', 'subset')] == 'small']

    # On construit un DataFrame simplifié
    df = pd.DataFrame({
        "track_id": subset.index,
        "split": subset[('set', 'split')],
        "genre_top": subset[('track', 'genre_top')],
    })

    # On enlève les morceaux sans genre_top
    df = df.dropna(subset=["genre_top"])

    # Encodage des genres en entiers
    genres = sorted(df["genre_top"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    df["genre_id"] = df["genre_top"].map(genre_to_idx)

    print("Genres trouvés :")
    for g, idx in genre_to_idx.items():
        count = (df["genre_top"] == g).sum()
        print(f"  {idx:2d} -> {g:20s} ({count} pistes)")

    return df, genre_to_idx


# =========================
# 3. Chemins vers les fichiers audio
# =========================

def track_id_to_path(track_id: int) -> Path:
    """
    Convertit un track_id (int) en chemin vers le fichier mp3 dans fma_small.
    Ex : 2 -> "000/000002.mp3"
    """
    track_id_str = f"{track_id:06d}"   # zero-pad sur 6 chiffres
    folder = track_id_str[:3]          # ex: "000", "001", ...
    return AUDIO_DIR / folder / f"{track_id_str}.mp3"


def add_audio_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'audio_path' au DataFrame.
    """
    print("Construction des chemins audio ...")
    df["audio_path"] = df["track_id"].apply(track_id_to_path)

    # Optionnel : vérifier les fichiers manquants
    missing_mask = ~df["audio_path"].apply(lambda p: p.exists())
    missing = df[missing_mask]

    if len(missing) > 0:
        print(f"Attention : {len(missing)} fichiers audio manquants.")
        # On peut les filtrer directement
        df = df[~missing_mask].reset_index(drop=True)
    else:
        print("Tous les fichiers audio existent.")

    return df


# =========================
# 4. Calcul du melspectrogramme
# =========================

def compute_mel(path: Path) -> np.ndarray:
    """
    Calcule un melspectrogramme normalisé pour un fichier audio donné.
    - Mel-spectrogramme (N_MELS x T)
    - conversion en dB
    - normalisation (z-score) par piste
    - padding/troncature à MAX_FRAMES
    """
    # charge le signal audio
    y, sr = librosa.load(path, sr=SR, mono=True)

    # melspectrogramme
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    # en dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # normalisation par piste
    mean = S_db.mean()
    std = S_db.std() + 1e-8
    S_norm = (S_db - mean) / std

    # padding / troncature sur l'axe temporel (axis=1)
    if S_norm.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - S_norm.shape[1]
        S_padded = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_padded = S_norm[:, :MAX_FRAMES]

    # float32 pour PyTorch
    return S_padded.astype(np.float32)


def compute_and_save_all_mels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque piste :
    - calcule le mel
    - le sauvegarde dans data/mels/{track_id:06d}.npy
    - enregistre le chemin (mel_path) dans le DataFrame

    Si une piste provoque une erreur, elle est ignorée.
    """
    print("Calcul des melspectrogrammes et sauvegarde en .npy ...")

    mel_paths = []
    valid_track_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["audio_path"]
        track_id = int(row["track_id"])
        mel_path = MELS_DIR / f"{track_id:06d}.npy"

        try:
            if not mel_path.exists():
                mel = compute_mel(audio_path)
                np.save(mel_path, mel)

            mel_paths.append(mel_path)
            valid_track_ids.append(track_id)

        except Exception as e:
            print(f"[ERREUR] {audio_path} : {e}")
            # on ignore ce morceau, on ne l'ajoute pas à la liste

    # on restreint le df aux pistes qui ont bien un mel
    df = df[df["track_id"].isin(valid_track_ids)].reset_index(drop=True)
    df["mel_path"] = mel_paths

    print(f"Nombre final de pistes avec mel : {len(df)}")
    return df


# =========================
# 5. Main
# =========================

def main():
    # Vérifications de base
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"AUDIO_DIR n'existe pas : {AUDIO_DIR}")
    if not META_DIR.exists():
        raise FileNotFoundError(f"META_DIR n'existe pas : {META_DIR}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv introuvable à : {tracks_path}")

    # 1) métadonnées + genres
    df, genre_to_idx = load_metadata(tracks_path)

    # 2) chemins audio
    df = add_audio_paths(df)

    # 3) calcul & sauvegarde des mels
    df = compute_and_save_all_mels(df)

    # 4) sauvegarde du CSV final
    output_csv = DATA_DIR / "features_mels.csv"
    df.to_csv(output_csv, index=False)
    print(f"CSV des features sauvegardé dans : {output_csv}")

    # 5) sauvegarde du mapping genre -> id (pour plus tard)
    genre_json = DATA_DIR / "genre_to_idx.json"
    with open(genre_json, "w", encoding="utf-8") as f:
        json.dump(genre_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Mapping des genres sauvegardé dans : {genre_json}")

    print("Préparation terminée ✅")


if __name__ == "__main__":
    main()
