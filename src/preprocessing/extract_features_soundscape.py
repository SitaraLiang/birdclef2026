import pandas as pd
import numpy as np
import librosa
import os
import concurrent.futures
import warnings
from extract_features_audio import get_feature_names, extract_features_from_chunk
from tqdm import tqdm
import argparse


def time_to_seconds(time_val):
    """Convertit '00:00:05' ou '00:05' en secondes pures."""
    if isinstance(time_val, (int, float)):
        return time_val
    parts = str(time_val).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(time_val)


def process_file_segments(args):
    """Charge un fichier audio complet, extrait les segments annotés."""
    warnings.filterwarnings("ignore")  #supprime le warning tonnetz dansles workers
 
    SR = 32000
    filename, group_df, audio_dir = args
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        return []
    try:
        y, sr = librosa.load(file_path, sr=SR)
    except Exception:
        return []

    results = []
    feature_names = get_feature_names()

    for _, row in group_df.iterrows():
        start_sec = row['start']
        end_sec   = row['end']
        labels    = row['primary_label']  # colonne dans train_soundscapes_labels.csv

        start_sample = int(start_sec * SR)
        end_sample   = int(end_sec * SR)
        chunk = y[start_sample:end_sample]

        features = extract_features_from_chunk(chunk, SR)

        if features is not None:
            data_row = {
                'filename': filename,
                'start':    start_sec,
                'end':      end_sec,
                'birds':    labels  
            }
            for name, val in zip(feature_names, features):
                data_row[name] = val
            results.append(data_row)

    return results


def main():
    print("=== EXTRACTION MULTI-CŒURS SOUNDSCAPES ===")
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', "--soundscape_labels", type=str, default="../../data/train_soundscapes_labels.csv")
    parser.add_argument('-sp', "--soundscape_dir",    type=str, default="../../data/train_soundscapes/")
    parser.add_argument('-o',  "--output",            type=str, default="X_val.csv")
    args = parser.parse_args()

    if not os.path.exists(args.soundscape_labels):
        print(f"Erreur : fichier {args.soundscape_labels} introuvable.")
        return

    df_labels = pd.read_csv(args.soundscape_labels)
    print(f"{len(df_labels)} segments de 5s trouvés dans le CSV.")

    df_labels['start'] = df_labels['start'].apply(time_to_seconds)
    df_labels['end']   = df_labels['end'].apply(time_to_seconds)

    grouped = df_labels.groupby('filename')
    tasks   = [(filename, group, args.soundscape_dir) for filename, group in grouped]

    all_data = []
    max_cores = max(1, os.cpu_count() - 1)
    print(f"Lancement sur {max_cores} cœurs CPU...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = list(tqdm(executor.map(process_file_segments, tasks),
                            total=len(tasks), desc="Extraction Soundscapes"))

    for sublist in results:
        all_data.extend(sublist)

    df_val = pd.DataFrame(all_data)
    df_val.to_csv(args.output, index=False)
    print(f"\nSuccès ! {len(df_val)} lignes générées dans '{args.output}'.")

if __name__ == "__main__":
    main()