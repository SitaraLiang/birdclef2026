import pandas as pd
import os
import argparse

def combine_datasets(audio_csv, soundscape_csv, output_csv):
    print("Début de la fusion des datasets")
    
    df_audio = pd.read_csv(audio_csv)
    df_soundscape = pd.read_csv(soundscape_csv)
    
    print(f"Audio : {len(df_audio)} lignes | Soundscape : {len(df_soundscape)} lignes")

    df_audio_clean = df_audio.copy()
    df_audio_clean['final_labels'] = df_audio_clean['all_labels'].fillna("")

    df_soundscape_clean = df_soundscape.copy()
    df_soundscape_clean['final_labels'] = df_soundscape_clean['birds'].str.replace(';', ' ')
    
    feature_cols = [c for c in df_audio.columns if 'mean' in c or 'std' in c]
    
    # Colonnes qu'on souhaite garder
    meta_cols = ['filename', 'final_labels']
    
    full_df = pd.concat([
        df_audio_clean[meta_cols + feature_cols],
        df_soundscape_clean[meta_cols + feature_cols]
    ], axis=0, ignore_index=True)

    full_df['final_labels'] = full_df['final_labels'].str.strip().str.replace(r'\s+', ' ', regex=True)

    full_df.to_csv(output_csv, index=False)
    
    print(f"Fusion terminée")
    print(f"Dataset final : {len(full_df)} lignes sauvegardées dans {output_csv}")
    print(f"Nombre de features : {len(feature_cols)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionne les datasets audio et soundscape en un seul CSV")
    parser.add_argument('--audio_csv', type=str, default='output/X_audio_features.csv', help='Chemin vers le CSV des features audio')
    parser.add_argument('--soundscape_csv', type=str, default='output/X_soundscape_features.csv', help='Chemin vers le CSV des features soundscape')
    parser.add_argument('--output_csv', type=str, default='output/final_features.csv', help='Chemin de sortie pour le dataset fusionné')
    args = parser.parse_args()

    if os.path.exists(args.audio_csv) and os.path.exists(args.soundscape_csv):
        combine_datasets(args.audio_csv, args.soundscape_csv, args.output_csv)
    else:
        print("Erreur: L'un des fichiers d'entrée est manquant.")