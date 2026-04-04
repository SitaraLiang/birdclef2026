import pandas as pd
import librosa
import os
import argparse
from tqdm import tqdm

def get_duration(row, base_path):
    """Helper to locate file and return duration in seconds."""
    # Try the two most likely path combinations
    paths_to_try = [
        os.path.join(base_path, row['primary_label'], row['filename']),
        os.path.join(base_path, row['filename'])
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                return librosa.get_duration(path=path)
            except Exception:
                return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Augment BirdCLEF metadata with file durations.")
    
    parser.add_argument("--input", type=str, default="data/train.csv", 
                        help="Path to the original train.csv")
    parser.add_argument("--audio_dir", type=str, default="data/train_audio/", 
                        help="Path to the root directory containing species folders")
    parser.add_argument("--output", type=str, default="data/train_metadata.csv", 
                        help="Path to save the enhanced CSV")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    print(f"Processing {len(df)} files from {args.audio_dir}...")

    tqdm.pandas(desc="Calculating Durations")
    df['duration'] = df.progress_apply(lambda row: get_duration(row, args.audio_dir), axis=1)

    initial_count = len(df)
    df_clean = df.dropna(subset=['duration'])
    missing = initial_count - len(df_clean)
    
    if missing > 0:
        print(f"Warning: {missing} files were missing or corrupted and were skipped.")

    df_clean.to_csv(args.output, index=False)
    print(f"Success! Enhanced metadata saved to: {args.output}")

if __name__ == "__main__":
    main()