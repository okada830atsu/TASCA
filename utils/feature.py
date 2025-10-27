
import os
import numpy as np
import librosa
from glob import glob
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()  # Traditional z-score normalization

    def simple_mfcc(self, data, sr, n_mfcc, n_fft, hop_length):
        features = []
        for y in data:
            #  MFCC extraction with different window sizes
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            
            feature = []
            # MFCC statistics (multiple statistics for better representation)
            feature.extend(np.mean(mfcc, axis=1))      # Mean
            features.append(feature)

        features = np.array(features, dtype=np.float32)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features

    def complex_mfcc(self, data, sr, n_mfcc, n_fft, hop_length):
        features = []
        for y in data:
            #  MFCC extraction with different window sizes
            mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            #mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=4096, hop_length=hop_length)
            
            # Additional spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            feature = []
            # MFCC statistics (multiple statistics for better representation)
            feature.extend(np.mean(mfcc1, axis=1))      # Mean
            feature.extend(np.std(mfcc1, axis=1))       # Standard deviation
            feature.extend(np.max(mfcc1, axis=1))       # Maximum
            feature.extend(np.min(mfcc1, axis=1))       # Minimum
            feature.extend(np.median(mfcc1, axis=1))    # Median
            
            # Second MFCC set with different parameters
            #feature.extend(np.mean(mfcc2, axis=1))
            #feature.extend(np.std(mfcc2, axis=1))
        
            # Spectral features
            feature.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
        
            # Delta and Delta-Delta features
            delta_mfcc  = librosa.feature.delta(mfcc1)
            delta2_mfcc = librosa.feature.delta(mfcc1, order=2)
            feature.extend(np.mean(delta_mfcc, axis=1))
            feature.extend(np.mean(delta2_mfcc, axis=1))

            features.append(feature)

        features = np.array(features, dtype=np.float32)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features