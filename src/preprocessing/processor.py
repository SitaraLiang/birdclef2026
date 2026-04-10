import librosa
import numpy as np
import os
import argparse

class AudioProcessor:
    def __init__(self, sr=32000, target_duration=5, max_chunks=12, min_duration=1.0):
        self.sr = sr
        self.target_duration = target_duration
        self.target_samples = sr * target_duration
        self.max_chunks = max_chunks
        self.min_duration = min_duration

    def process_file(self, file_path):
        """
        Loads, resamples, and handles short/long audio files.
        Returns a list of processed audio chunks (numpy arrays).
        Sample Rate (sr): 32,000, every 1 second of bird song is captured by 32,000 digital data points.
        Duration: 5 seconds -> 160,000 samples per chunk.
        """
        # 1. Load & Resample
        try:
            y, _ = librosa.load(file_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

        file_len = len(y)
        duration = file_len / self.sr
        
        # 2. Reject ultra-short "garbage" files (< 1s)
        if duration < self.min_duration:
            return []

        # 3. Handle Short Files (1s to 5s) -> Wrap them
        if file_len < self.target_samples:
            y_wrapped = np.tile(y, int(np.ceil(self.target_samples / file_len)))[:self.target_samples]
            return [y_wrapped]
        
        # 4. Handle Long Files -> Chunking
        chunks = []
        for i in range(0, file_len, self.target_samples):
            # Stop if we hit our diversity limit (e.g., 12 chunks/60 seconds)
            if len(chunks) >= self.max_chunks:
                break
                
            chunk = y[i:i + self.target_samples]
            
            # Full 5s chunk
            if len(chunk) == self.target_samples:
                chunks.append(chunk)
            
            # Significant tail (> 2s) -> Wrap it instead of padding
            elif len(chunk) > (self.sr * 2):
                chunk_wrapped = np.tile(chunk, int(np.ceil(self.target_samples / len(chunk))))[:self.target_samples]
                chunks.append(chunk_wrapped)
        
        return chunks

