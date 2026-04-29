import pandas as pd
import numpy as np
import librosa
import os
import argparse
from tqdm import tqdm
import concurrent.futures
import warnings
from processor import AudioProcessor

def get_feature_names():
    names = []
    # --- Means ---
    names += [f'mfcc_{i}_mean' for i in range(13)]         # 13
    names += [f'delta_{i}_mean' for i in range(13)]        # 13
    names += [f'delta2_{i}_mean' for i in range(13)]       # 13
    names += ['zcr_mean', 'centroid_mean', 'rolloff_mean'] # 3
    names += [f'chroma_{i}_mean' for i in range(12)]       # 12
    names += [f'mel_{i}_mean' for i in range(64)]          # 64
    names += [f'contrast_{i}_mean' for i in range(7)]      # 7 (n_bands=6 → 7 valeurs)
    names += ['bandwidth_mean']                             # 1
    names += [f'tonnetz_{i}_mean' for i in range(6)]       # 6
    # --- Stds ---
    names += [f'mfcc_{i}_std' for i in range(13)]          # 13
    names += [f'delta_{i}_std' for i in range(13)]         # 13
    names += [f'delta2_{i}_std' for i in range(13)]        # 13
    names += ['zcr_std', 'centroid_std', 'rolloff_std']    # 3
    names += [f'chroma_{i}_std' for i in range(12)]        # 12
    names += [f'mel_{i}_std' for i in range(64)]           # 64
    names += [f'contrast_{i}_std' for i in range(7)]       # 7
    names += ['bandwidth_std']                              # 1
    names += [f'tonnetz_{i}_std' for i in range(6)]        # 6

    # Total : 132 means + 132 stds = 264 features
    assert len(names) == 264, f"Erreur : {len(names)} features au lieu de 264"
    return names


def extract_features_from_chunk(y, sr=32000):
    try:
        # Pré-traitement anti-bruit
        y_harmonic, _ = librosa.effects.hpss(y)
        y = librosa.util.normalize(y_harmonic)

        # --- Features existantes ---
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # --- Nouvelles features ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)           # (12, T)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,fmin=500,fmax=12000)  # fréquences oiseaux généralement entre 500 et 12000 des recherches que j'ai vu, ça devrait aider à isoler passivement ces sons 
        mel_db = librosa.power_to_db(mel)     # (64, T) features semblant très importantes pour le model 
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr,n_bands=6, fmin=100.0) # features très importantes pour le model 
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # (1, T)

        #à noter que si ces nouvelles features te donne de moins bons resultats, tu pourras juste les masquer dans le df pour retrouver tes anciens resultats normalement

        with warnings.catch_warnings(): # juste enlever des warnings chiants dans le terminal, 
            warnings.simplefilter("ignore")
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)  # (6, T)

        features = np.concatenate([
            # means
            np.mean(mfcc, axis=1),         # 13
            np.mean(delta_mfcc, axis=1),   # 13
            np.mean(delta2_mfcc, axis=1),  # 13
            [np.mean(zcr)],                # 1
            [np.mean(centroid)],           # 1
            [np.mean(rolloff)],            # 1
            np.mean(chroma, axis=1),       # 12
            np.mean(mel_db, axis=1),       # 64
            np.mean(contrast, axis=1),     # 7
            [np.mean(bandwidth)],          # 1
            np.mean(tonnetz, axis=1),      # 6

            # stds
            np.std(mfcc, axis=1),          # 13
            np.std(delta_mfcc, axis=1),    # 13
            np.std(delta2_mfcc, axis=1),   # 13
            [np.std(zcr)],                 # 1
            [np.std(centroid)],            # 1
            [np.std(rolloff)],             # 1
            np.std(chroma, axis=1),        # 12
            np.std(mel_db, axis=1),        # 64
            np.std(contrast, axis=1),      # 7
            [np.std(bandwidth)],           # 1
            np.std(tonnetz, axis=1),       # 6
        ])
        return features
        # 264 features au total
    except Exception:
        return None


def process_single_file(args):
    """Fonction exécutée par chaque Worker (Cœur CPU) en parallèle."""
    warnings.filterwarnings("ignore")
    row, audio_dir = args

    audio_processor = AudioProcessor(sr=32000, target_duration=5, max_chunks=12)
    file_path = os.path.join(audio_dir, row['filename'])

    if not os.path.exists(file_path):
        return []

    chunks = audio_processor.process_file(file_path)
    lignes_extraites = []

    feature_names = get_feature_names()

    for chunk in chunks:
        features = extract_features_from_chunk(chunk)
        if features is not None:
            ligne_donnee = {
                'primary_label': row['primary_label'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'filename': row['filename']
            }
            for nom, val in zip(feature_names, features):
                ligne_donnee[nom] = val
            lignes_extraites.append(ligne_donnee)

    return lignes_extraites


def main():
    print("=== EXTRACTION MASSIVE MULTI-CŒURS ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  type=str, default="../../data/train_metadata.csv")
    parser.add_argument("--audio_dir", type=str, default="../../data/train_audio/")
    parser.add_argument("--output",    type=str, default="X_train.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    tasks = [(row, args.audio_dir) for _, row in df.iterrows()]
    donnees_finales = []

    # ----- Comme avant, partie servant à raccourcir le délai de traitement des 191k lignes avec plus de 200 features.

    max_cores = max(1, os.cpu_count() - 1)
    print(f"Lancement de l'extraction sur {max_cores} cœurs CPU...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = list(tqdm(executor.map(process_single_file, tasks),
                            total=len(tasks), desc="Processing audio"))

    for sous_liste in results:
        donnees_finales.extend(sous_liste)

    df_features = pd.DataFrame(donnees_finales)
    df_features.to_csv(args.output, index=False)

    print(f"\nTerminé ! {len(df_features)} morceaux de 5 secondes extraits.")
    print(f"Fichier '{args.output}' — 264 features par segment.")

if __name__ == "__main__":
    main()
