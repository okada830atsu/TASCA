
import os
import numpy as np
import librosa
from glob import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from umap.umap_ import UMAP
import argparse
import datetime


import numpy as np
from scipy.optimize import linear_sum_assignment

def load_data(data_dir, keystroke_count, sample_rate, label):

    data   = []
    labels = []

    file_paths = sorted(glob(os.path.join(data_dir, "*.wav")))[:keystroke_count]
    
    print(f"Extracting  features from {len(file_paths)} files...")
    
    for path in tqdm(file_paths, desc="Extracting features"):
        filename = os.path.basename(path)
        
        if label:
            labels.append(filename.split('_')[1].split('.')[0])
        else:
            labels.append(filename.split('.')[0])
        
        y, sr = librosa.load(path, sr=sample_rate)
        data.append(y)

    return np.array(data), np.array(labels)

def calculate_accuracy(predictions, true_labels):
    """
    クラスタリングの正解率を計算する関数。
    各クラスタと真のラベルの対応をハンガリアン法（最適割当）で求め、
    その最適対応に基づいて精度（Accuracy）を算出します。

    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        モデルのクラスタ割当結果（例: KMeansのラベルなど）
    true_labels : array-like of shape (n_samples,)
        各サンプルの真のラベル

    Returns
    -------
    accuracy : float
        0〜1の範囲の正解率（最適ラベル対応後）
    """

    # NumPy配列に変換し、安全のため整数型に
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_true = np.asarray(true_labels, dtype=np.int64)
    assert y_pred.size == y_true.size, "predictions と true_labels のサイズが一致しません"

    # クラスタ×ラベルの共起数行列を作成
    D = max(y_pred.max(), y_true.max()) + 1  # クラス数（予測・真値の最大値＋1）
    contingency = np.zeros((D, D), dtype=np.int64)
    for pred_label, true_label in zip(y_pred, y_true):
        contingency[pred_label, true_label] += 1

    # Hungarian algorithm（線形割当問題）で最適対応を求める
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)

    # 対応後の正解サンプル数を合計して精度を算出
    total_correct = contingency[row_ind, col_ind].sum()
    accuracy = total_correct / y_pred.size

    return accuracy

def save_confusion_matrix(predictions, true_labels, img_dir, method, reporter=None):

    """Save confusion matrix as heatmap (both numeric and label versions)"""
    # Ensure img directory exists
    os.makedirs(img_dir, exist_ok=True)
    
    # Convert labels to numeric for evaluation
    label_encoder = LabelEncoder()
    true_labels_numeric = label_encoder.fit_transform(true_labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels_numeric, predictions)
    
    # 2. Save label version (new)
    # Determine representative label for each cluster
    cluster_to_label = {}
    for cluster_id in range(cm.shape[1]):
        # Find samples assigned to this cluster
        cluster_mask = predictions == cluster_id
        if np.any(cluster_mask):
            # Get true labels for samples in this cluster
            cluster_true_labels = true_labels_numeric[cluster_mask]
            # Find most common true label (majority vote)
            most_common_label_idx = np.bincount(cluster_true_labels).argmax()
            representative_label = label_encoder.classes_[most_common_label_idx]
            cluster_to_label[cluster_id] = representative_label
        else:
            cluster_to_label[cluster_id] = '?'  # Empty cluster
    
    # Create complete label set (a-z + space) and initialize with zeros
    complete_labels = list('abcdefghijklmnopqrstuvwxyz') + [' ']
    new_cm = []
    new_cluster_labels = []
    
    for label in complete_labels:
        # Find clusters for this label
        cluster_ids = []
        for cluster_id, cluster_label in cluster_to_label.items():
            if cluster_label == label or (label == ' ' and cluster_label == 'space'):
                cluster_ids.append(cluster_id)
        
        if cluster_ids:
            # Sum columns for clusters with this label
            if len(cluster_ids) == 1:
                new_cm.append(cm[:, cluster_ids[0]])
            else:
                # Multiple clusters with same label - sum them
                combined_column = np.sum(cm[:, cluster_ids], axis=1)
                new_cm.append(combined_column)
        else:
            # No clusters for this label - create zero column
            new_cm.append(np.zeros(cm.shape[0], dtype=int))
        
        new_cluster_labels.append(label)
    
    new_cm = np.column_stack(new_cm)
    
    # Reorder rows to match complete label set (a-z + space)
    # Create mapping from label_encoder.classes_ to complete_labels order
    label_to_row = {label: i for i, label in enumerate(label_encoder.classes_)}
    
    # Create 27x27 matrix with proper row ordering
    final_cm = np.zeros((27, 27), dtype=int)
    
    for new_row_idx, label in enumerate(complete_labels):
        if label in label_to_row:
            old_row_idx = label_to_row[label]
            final_cm[new_row_idx, :] = new_cm[old_row_idx, :]
        elif label == ' ' and 'space' in label_to_row:
            # Handle space mapping from 'space' string to ' ' character
            old_row_idx = label_to_row['space']
            final_cm[new_row_idx, :] = new_cm[old_row_idx, :]
        # If label not in data, row remains zero
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(final_cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=new_cluster_labels,
                yticklabels=complete_labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix (Representative Labels) - {method.upper()} Clustering')
    plt.xlabel('Predicted Cluster (Representative Label)')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    filepath_labels = os.path.join(img_dir, f"confusion_matrix_labels_{method}.png")
    plt.savefig(filepath_labels, dpi=150, bbox_inches='tight')
    plt.close()
    
    if reporter:
        reporter.write(f"Confusion matrices saved: {filepath_labels}")
    
    print(f"Confusion matrix (labels) saved: {filepath_labels}")
    return filepath_labels

def evaluate_clustering(predictions, true_labels, le, img_dir, method, reporter):
    """Evaluate clustering performance using multiple metrics"""
    if reporter:
        reporter.write("=== Traditional K-means Clustering Evaluation ===")
    
    # Convert labels to numeric for evaluation
    true_labels_numeric = le.fit_transform(true_labels)
    
    # Calculate clustering metrics
    ari = adjusted_rand_score(true_labels_numeric, predictions)
    nmi = normalized_mutual_info_score(true_labels_numeric, predictions)
    
    # Calculate accuracy using optimal assignment (same as DEC evaluation)
    accuracy = calculate_accuracy(predictions, true_labels_numeric)
    
    results = {
        'accuracy': accuracy,
        'ari': ari,
        'nmi': nmi,
    }
    
    print(f"\nTraditional K-means Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    if reporter:
        reporter.write(f"K-means Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        reporter.write(f"Adjusted Rand Index: {ari:.4f}")
        reporter.write(f"Normalized Mutual Information: {nmi:.4f}")
    
    # Save confusion matrix
    save_confusion_matrix(predictions, true_labels, img_dir, "kmeans", reporter)
    
    return results


class TraditionalKMeans: 
    def __init__(self, n_clusters, img_dir, random_state=42):
        self.n_clusters = n_clusters
        self.img_dir = img_dir
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=1000)
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit_predict(self, features):
        """Fit K-means and return cluster predictions"""
        print(f"Performing traditional K-means clustering with {self.n_clusters} clusters...")
        self.labels_ = self.kmeans.fit_predict(features)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        print(f"K-means clustering completed")
        print(f"Inertia (within-cluster sum of squares): {self.kmeans.inertia_:.2f}")
        
        return self.labels_
 
class DECNet(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super(DECNet, self).__init__()
        
        #  encoder with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        #  decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, input_dim)
        )
        
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        #  clustering with temperature scaling
        q = 1.0 / (1.0 + torch.sum(torch.pow(encoded.unsqueeze(1) - self.clustering_layer, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return encoded, decoded, q

class CNN_DECNet(nn.Module):
    def __init__(self, input_channels, latent_dim, n_clusters, input_shape=(1, 64, 86)):
        super(CNN_DECNet, self).__init__()

        self.input_shape = input_shape  # e.g., (1, 64, 86)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # --- CNN Encoder ---
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=10, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --- flatten size calculation ---
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.encoder_cnn(dummy)
            self.flatten_dim = out.numel()

        # --- latent projection ---
        self.fc_enc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )

        # --- decoder (reverse process) ---
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.flatten_dim),
            nn.ReLU()
        )

        # store last feature-map shape
        self.feature_shape = out.shape[1:]  # (C, H, W)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # clustering layer
        self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_normal_(self.clustering_layer.data)

    def forward(self, x):
        z = self.encoder_cnn(x)
        z_flat = z.view(z.size(0), -1)
        latent = self.fc_enc(z_flat)

        h = self.fc_dec(latent)
        h = h.view(h.size(0), *self.feature_shape)
        x_recon = self.decoder_cnn(h)

        # resize reconstructed output to match input exactly (if needed)
        if x_recon.shape != x.shape:
            x_recon = F.interpolate(x_recon, size=x.shape[2:], mode='bilinear', align_corners=False)

        q = 1.0 / (1.0 + torch.sum(torch.pow(latent.unsqueeze(1) - self.clustering_layer, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return latent, x_recon, q
        # Encoder
        z = self.encoder_cnn(x)
        z = z.view(z.size(0), -1)
        latent = self.fc_enc(z)

        # Decoder
        h = self.fc_dec(latent)
        h = h.view(h.size(0), 256, 4, 4)
        x_recon = self.decoder_cnn(h)

        # DECクラスタ確率 q
        q = 1.0 / (1.0 + torch.sum(torch.pow(latent.unsqueeze(1) - self.clustering_layer, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return latent, x_recon, q

class DEC:
    def __init__(self, n_clusters, model, latent_dim, device, model_path, img_dir, reporter):
        self.n_clusters = n_clusters
        self.model = model
        self.latent_dim = latent_dim
        self.device = device
        self.model_path = model_path
        self.img_dir = img_dir
        self.reporter = reporter
        self.best_accuracy = 0.0



    def train_autoencoder(self, features, epochs, batch_size, learning_rate):
        """Train the autoencoder with learning rate scheduling or load saved model"""
        
        # Train new model
        self.reporter.write("Phase 1: Autoencoder Pre-training started")
        print(f"\nTraining Autoencoder... (will save to {self.model_path})")
        
        features_tensor = torch.from_numpy(features).to(self.device)
        dataset = TensorDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
        criterion = nn.MSELoss()
        
        final_loss = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for data in dataloader:
                batch_features = data[0]
                optimizer.zero_grad()
                _, decoded, _ = self.model(batch_features)
                loss = criterion(decoded, batch_features)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            final_loss = avg_epoch_loss
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'AE Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}')
                self.reporter.write(f"AE Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save the trained model
        print(f"\nSaving autoencoder model to {self.model_path}...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'final_loss': final_loss,
            'input_dim': features.shape[1],
            'latent_dim': self.latent_dim,
            'epochs': epochs,
            'learning_rate': learning_rate
        }, self.model_path)
        print("Model saved successfully!")
        self.reporter.write(f"Autoencoder model saved to {self.model_path} with final loss: {final_loss:.4f}")
    
    def initialize_clusters(self, features, label_ids, n_clusters):
        """ cluster initialization"""
        self.reporter.write("Phase 2: DEC Initialization & Fine-tuning started")
        print(f"\nInitializing  cluster centers (N_CLUSTERS={n_clusters})...")
        
        features_tensor = torch.from_numpy(features).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            encoded_features, _, _ = self.model(features_tensor)
        
        #  K-means with multiple initializations
        best_kmeans = None
        best_inertia = float('inf')
        
        for i in range(5):  # Try multiple initializations
            kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42+i, max_iter=500)
            kmeans.fit(encoded_features.cpu().numpy())
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
        
        y_pred_kmeans = best_kmeans.labels_
        self.model.clustering_layer.data = torch.tensor(best_kmeans.cluster_centers_).to(self.device)
        
        # Evaluate initial accuracy
        initial_acc = self._evaluate_clustering(y_pred_kmeans, label_ids, n_clusters)
        print(f"Initial  DEC accuracy: {initial_acc * 100:.2f}%")
        self.reporter.write(f"Initial  DEC accuracy: {initial_acc * 100:.2f}%")
        
        return y_pred_kmeans
         
    def train_dec(self, features, epochs, learning_rate, update_interval, tolerance):
        """DEC training"""
        print("\nStarting  DEC fine-tuning...")
        
        features_tensor = torch.from_numpy(features).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Get initial predictions
        self.model.eval()
        with torch.no_grad():
            _, _, q = self.model(features_tensor)
        y_pred_last = q.argmax(1).cpu().numpy()
        
        patience_counter = 0
        max_patience = 20
        
        # Disable dropout/BatchNorm updates during DEC fine-tuning
        self.model.eval()

        for epoch in range(epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, _, q = self.model(features_tensor)
                    p = self._target_distribution(q.data)
                
                y_pred = q.argmax(1).cpu().numpy()
                
                # Check for convergence (no accuracy check in predict mode)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                
                if epoch > 0 and delta_label < tolerance:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f'Convergence reached at epoch {epoch}. Delta: {delta_label:.6f}')
                        self.reporter.write(f"Convergence reached at epoch {epoch}.")
                        break
                else:
                    patience_counter = 0
            
            # Training step (use eval mode to keep dropout disabled while allowing gradients)
            self.model.train()
            _, _, q = self.model(features_tensor)
            loss = F.kl_div(q.log(), p, reduction='batchmean')
            
            if epoch % update_interval == 0:
                print(f"DEC Epoch [{epoch}/{epochs}]: loss = {loss.item():.6f}")
                self.reporter.write(f"DEC Epoch [{epoch}/{epochs}]: loss = {loss.item():.6f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_message = f" DEC training finished"
        self.reporter.write(final_message)
        print(final_message)
    
    def _target_distribution(self, q):
        """ target distribution computation"""
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
