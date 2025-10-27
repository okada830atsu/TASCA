
import os
import numpy as np
import librosa
from glob import glob
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

class AutoencoderFeatureExtractor:
    def __init__(self, data_dir, sample_rate, n_mfcc, mode):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mode = mode
        self.scaler = StandardScaler()  # Traditional z-score normalization

    
    def extract__features(self, keystroke_count):
        """Extract  MFCC features with multiple statistics"""
        features = []
        labels = []
        file_paths = sorted(glob(os.path.join(self.data_dir, "*.wav")))[:keystroke_count]
        
        print(f"Extracting  features from {len(file_paths)} files...")
        
        for path in tqdm(file_paths, desc="Extracting features"):
            filename = os.path.basename(path)
            
            # Extract label based on mode
            if self.mode == 'evaluate':
                # Evaluate mode: extract label from filename like "a_100.wav"
                #label = filename.split('_')[0]
                label = filename.split('_')[1].split('.')[0]
            else:
                # Predict mode: use filename without extension like "0001.wav" -> "0001"
                label = filename.split('.')[0]
            
            y, sr = librosa.load(path, sr=self.sample_rate)
            
            #  MFCC extraction with different window sizes
            mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=8192, hop_length=512)
            mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=4096, hop_length=512)
            
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
            feature.extend(np.mean(mfcc2, axis=1))
            feature.extend(np.std(mfcc2, axis=1))
        
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
            labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        
        # Apply standardization (z-score normalization)
        features = self.scaler.fit_transform(features)
        
        return features, np.array(labels)
    
    def extract_features(self, feature_file, label_file, keystroke_count):

        print("Extracting  features from audio files...")
        features, labels = self.extract__features(keystroke_count)
        np.save(feature_file, features)
        np.save(label_file, labels)
        print(" features and labels saved to .npy files.")
        
        return features, labels
    
class KmeansFeatureExtractor:
    def __init__(self, data_dir, sample_rate, n_mfcc, mode):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mode = mode
        self.scaler = StandardScaler()  # Traditional z-score normalization

    
    def extract__features(self, keystroke_count):
        """Extract  MFCC features with multiple statistics"""
        features = []
        labels = []
        file_paths = sorted(glob(os.path.join(self.data_dir, "*.wav")))[:keystroke_count]
        
        print(f"Extracting  features from {len(file_paths)} files...")
        
        for path in tqdm(file_paths, desc="Extracting features"):
            filename = os.path.basename(path)
            
            # Extract label based on mode
            if self.mode == 'evaluate':
                # Evaluate mode: extract label from filename like "a_100.wav"
                #label = filename.split('_')[0]
                label = filename.split('_')[1].split('.')[0]
            else:
                # Predict mode: use filename without extension like "0001.wav" -> "0001"
                label = filename.split('.')[0]
            
            y, sr = librosa.load(path, sr=self.sample_rate)
            
            #  MFCC extraction with different window sizes
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=8192, hop_length=512)
            
            feature = []
            feature.extend(np.mean(mfcc, axis=1))      # Mean
        
            features.append(feature)
            labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        
        # Apply standardization (z-score normalization)
        features = self.scaler.fit_transform(features)
        
        return features, np.array(labels)
    
    def extract_features(self, feature_file, label_file, keystroke_count):

        print("Extracting  features from audio files...")
        features, labels = self.extract__features(keystroke_count)
        np.save(feature_file, features)
        np.save(label_file, labels)
        print(" features and labels saved to .npy files.")
        
        return features, labels    