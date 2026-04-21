import pandas as pd
import numpy as np
import librosa
import os
import concurrent.futures
from extract_features_audio import get_feature_names, extract_features_from_chunk
from tqdm import tqdm
import argparse

# fonction utilitaire pout récupérer les secondes du fichier train_soundscapes_labels.py
def time_to_seconds(time_val):
    """Convertit '00:00:05' ou '00:05' en secondes pures."""

    if isinstance(time_val, (int, float)): # par mesure de sécuritéé
        return time_val

    parts= str(time_val).split(':')
    if len(parts) == 3: #format HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    elif len(parts) == 2: # format MM:SS (au cas où le format changerait)
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(time_val)

def process_file_segments(args):
    
    #Charge un fichier audio complet, puis découpe uniquement les segments (start/ end) annotés dans le csv
    SR= 32000
    filename, group_df, audio_dir = args
    file_path = os.path.join(audio_dir, filename)

    if not os.path.exists(file_path):
        return []
    try:
        # charge les 60 secondes en une seule fois (c'est plus rapide )
        y, sr = librosa.load(file_path, sr=SR)
    except Exception:
        return []

    results = []
    feature_names = get_feature_names() # la fonction faite dans extract_features_audio qui donne les noms de colonnes pour le csv final

    # on boucle sur les annotations de temps specifiques à un fichier précis
    for inutile, row in group_df.iterrows():
        start_sec = row['start']
        end_sec = row['end']

        labels = row['primary_label'] # Ex: "bird1;bird2" ou "nocall"

        # conversion des secondes en indices de tableau numpy
        start_sample = int( start_sec * SR)
        end_sample = int(end_sec * SR)

        #on extrait la  bonne tranche 
        chunk = y[ start_sample :  end_sample]

        """
        # Sécurité : Si le chunk fait moins de 5s, on complète avec du silence (padding)
        target_len = SR * 5
        if len(chunk) < target_len:
            print("TEEEEEEST : bizarre, un chunk fait moins de 5 secondes ")
            chunk = np.pad(chunk, (0, target_len - len(chunk)))
            print("TEEEEEEST 2:",chunk )
        """

        #on extrait les features avec la même méthode que dans le fichier extract_features_audio
        features = extract_features_from_chunk(chunk, SR)
        
        if features is not None:
            data_row = {
                'filename': filename,
                'start':  start_sec,
                'end': end_sec,
                'birds': labels
            }
            # je fais une boucle pour attribuer chaque feature à son nom de feature dans le csv 
            for name, val in zip(feature_names, features):
                data_row[name] = val
            
            results.append(data_row)

    # à noter que dans les fichiers du dossier train_soundscapes, il n'y a pas la les features de longitude et de latitude
    # je crois que dans le nom d'un fichier du dossier train_sounscape, on a une parti qui donne le site, meme, si c'est moins precis, c'est peut etre interessant à capturer pour remplacer les features de geographie
    # quelque chose à pousser si je vois que ces informations sont vraiment essentiel au model 
    # bref, je vais voir ça un peu plus tard   
    return results

def main():

    print("=== EXTRACTION RApide  MULTI CŒURS ===")
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl',"--soundscape_labels", type=str, default="../../data/train_soundscapes_labels.csv")
    parser.add_argument('-sp',"--soundscape_dir", type=str, default="../../data/train_soundscapes/")
    parser.add_argument('-o', "--output", type=str, default="X_val.csv")
    args = parser.parse_args()

    AUDIO_DIR = args.soundscape_dir
    LABELS_FILE = args.soundscape_labels
    OUTPUT_FILE = args.output

    print("=== CRÉATION DU DATASET DE VALIDATION (Soundscapes) ===")
    
    if not os.path.exists(LABELS_FILE):
        print(f"Erreur : Fichier {LABELS_FILE} introuvable.")
        return

    df_labels = pd.read_csv(LABELS_FILE)
    print(f"{len(df_labels)} segments de 5s à extraire trouvés dans le CSV.")

    df_labels['start'] = df_labels['start'].apply(time_to_seconds)
    df_labels['end'] = df_labels['end'].apply(time_to_seconds)
    
    # On groupe les annotations par fichier pour ne charger chaque .ogg qu'une seule fois
    grouped = df_labels.groupby('filename')
    tasks = [(filename, group, AUDIO_DIR) for filename, group in grouped]
    
    all_data = []
    max_cores = os.cpu_count() -1
    
    #lancement du multiprocessing pour optimiser la rapidité de traitemennt 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        results = list(tqdm(executor.map(process_file_segments, tasks), total=len(tasks), desc="Extraction Soundscapes"))
    
    for sublist in results:
        all_data.extend(sublist)

    df_val = pd.DataFrame(all_data)
    df_val.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccès ! {len(df_val)} lignes générées dans {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()