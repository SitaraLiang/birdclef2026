import ast

import pandas as pd
import numpy as np
import librosa
import os
import argparse
from tqdm import tqdm
import concurrent.futures
from processor import AudioProcessor

# README !!

# pour cette première version d'extraction des features, on en prend au total 84.
# MFCC donne un vecteur de 13 coefficients
# Pour chacun de ces coefficients on a 2 features, la moyenne et l'ecart type
# => explication : Le script passe à travers les 5 secondes de chaque extrait, cependant le MFCC prend une fenetre de 25 milliseconde en analyse , on prend donc la moyenne et l'ecart type des resultats de chauqe fenetre pour le son de 5seconde.
# j'ai choisi d'enregistrer aussi l'ecart type, ça me semblait interessant à prendre

#On suit la meme logique pour delta mfcc, et delta2 mfcc.., ajouté aux features centroid, zrc et rolloff, ça nous donne au cumulé les 84 features.

#À noter que le fichier analyse_features permet de réveler, les features potentiellement les plus importantes pour le futur model.

def get_feature_names():
    """Génère les noms exacts pour les 84 colonnes extraites."""
    names = []
    # mean
    names += [f'mfcc_{i}_mean' for i in range(13)]
    names += [f'delta_{i}_mean' for i in range(13)]
    names += [f'delta2_{i}_mean' for i in range(13)]
    names += ['zcr_mean', 'centroid_mean', 'rolloff_mean']
    
    # stds
    names += [f'mfcc_{i}_std' for i in range(13)]
    names += [f'delta_{i}_std' for i in range(13)]
    names += [f'delta2_{i}_std' for i in range(13)]
    names += ['zcr_std', 'centroid_std', 'rolloff_std']
    
    return names

def extract_features_from_chunk(y, sr=32000):
    """Calcule les caractéristiques sur un morceau audio de 5 secondes."""
    try:

        zcr = librosa.feature.zero_crossing_rate(y) #combien de fois l'oscillation de la frequence passe par 0

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1), 
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1),
            [np.mean(zcr)], [np.mean(centroid)], [np.mean(rolloff)],
            np.std(mfcc, axis=1),
            np.std(delta_mfcc, axis=1), 
            np.std(delta2_mfcc, axis=1),
            [np.std(zcr)], [np.std(centroid)], [np.std(rolloff)]
        ])
        return features
    except Exception:
        return None

def process_single_file(args):
    """Fonction exécutée par chaque Worker (Cœur CPU) en parallèle."""
    # On dépaquette les arguments
    row, audio_dir = args

    #on  instancie le processeur à l'intérieur du worker pour evviter les conflits de mémoire
    audio_processor = AudioProcessor(sr=32000, target_duration=5, max_chunks=12)
    file_path = os.path.join(audio_dir, row['filename'])
    
    if not os.path.exists(file_path):
        return []
        
    chunks = audio_processor.process_file(file_path)
    lignes_extraites = []
    feature_names = get_feature_names()

    # On combine primary et secondary en une seule liste propre
    primary = row['primary_label']
    try:
        # ast.literal_eval transforme "['a', 'b']" en ['a', 'b']
        secondary = ast.literal_eval(row['secondary_labels']) if isinstance(row['secondary_labels'], str) else []
    except (ValueError, SyntaxError):
        secondary = []

    all_labels = " ".join([primary] + secondary).strip()


    for chunk in chunks:
        features = extract_features_from_chunk(chunk)
        if features is not None:
            #dictionnaire de base avec les infos kaggle
            ligne_donnee = {
                'primary_label': primary,
                'secondary_labels': row['secondary_labels'],
                'all_labels': all_labels,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'filename': row['filename']
            }
            #assigne les 84 valeurs aux noms de choisi
            for nom, val in zip(feature_names, features):
                ligne_donnee[nom] = val
                
            lignes_extraites.append(ligne_donnee)
            
    return lignes_extraites

def main():
    print("=== EXTRACTION MASSIVE MULTI-CŒURS ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/train_metadata.csv")
    parser.add_argument("--audio_dir", type=str, default="data/train_audio/")
    parser.add_argument("--output", type=str, default="output/X_audio_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    
    # Pour optimiser le temps de traitement, il nous est nécessaire d'utiliser du multithread
    # On passe de plusieurs longues heures de traitement, à moins d'une heure

    tasks = [(row, args.audio_dir) for index, row in df.iterrows()]
    donnees_finales = []
    
    # Lancement du multiprocessing
    #faire attention, ça utilise tous les coeurs -1 du cpu, à ne faire que sur le cluster de la fac
    max_cores = os.cpu_count() -1
    print(f" Lancement de l'extraction sur {max_cores} cœurs CPU...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        # map exécute process_single_file sur chaque élément de "tasks" en parallèle et tqdm pour l'affichage continu de la barre de progression dans le terminal
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing audio"))
    
    # 'results' est une liste de listes. On l'applatit en une seule grande liste
    for sous_liste in results:
        donnees_finales.extend(sous_liste)

    df_features = pd.DataFrame(donnees_finales)
    df_features.to_csv(args.output, index=False)
    
    print(f"\nTerminé ! {len(df_features)} morceaux de 5 secondes ont été extraits.")
    print(f"Le fichier '{args.output}' contient maintenant de vrais noms de colonnes comme 'mfcc_X_mean' !")

if __name__ == "__main__":
    main()