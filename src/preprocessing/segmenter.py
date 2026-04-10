import librosa
import numpy as np
import argparse
import os

class SoundscapeProcessor:
    def __init__(self, sr=32000, duration=5):
        self.sr = sr
        self.duration = duration
        self.target_samples = sr * duration

    def process_full_file(self, file_path):
        """Slices a long soundscape into 12 consecutive 5s chunks."""
        # 1. Load the whole file (usually 1 min)
        y, _ = librosa.load(file_path, sr=self.sr)
        
        file_len = len(y)
        chunks = []
        
        # 2. Step through the file in 5-second increments
        for i in range(0, file_len, self.target_samples):
            chunk = y[i:i + self.target_samples]
            
            # If the chunk is exactly 5s, keep it
            if len(chunk) == self.target_samples:
                chunks.append(chunk)
            # If there's a tiny bit left at the end, pad it to 5s
            elif len(chunk) > 0:
                chunk = np.pad(chunk, (0, self.target_samples - len(chunk)), mode='constant')
                chunks.append(chunk)
                
        return chunks
