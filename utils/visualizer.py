

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
try:
    # 通常のインポート（umap-learn >= 0.5系など）
    from umap import UMAP
except ImportError:
    # 古い構成ではクラスをモジュールから取り出す
    import umap.umap_ as umap_module
    UMAP = umap_module.UMAP


def visualize_clustering(img_dir, model, features, labels, device, title_suffix=""):
    """Create interactive UMAP visualization of clustering results using Plotly"""
    
    print("Creating interactive UMAP visualization...")
    
            # Apply UMAP
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='cosine'
    )
    
    if(model==None):
        embedding = reducer.fit_transform(features)
    else:
        # Get latent representations
        features_tensor = torch.from_numpy(features).to(device)
        model.eval()
        with torch.no_grad():
            encoded_features, _, _ = model(features_tensor)
            latent_features = encoded_features.cpu().numpy()
        embedding = reducer.fit_transform(latent_features)
    
    # Create color palette for 27 classes
    unique_labels = sorted(np.unique(labels))
    colors = colors = px.colors.qualitative.G10 + px.colors.qualitative.Light24
    
    # Create interactive scatter plot
    fig = go.Figure()
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            name=f"'{label}' ({np.sum(mask)} samples)",
            hovertemplate=f"<b>Label:</b> {label}<br>" +
                        "<b>UMAP-1:</b> %{x:.2f}<br>" +
                        "<b>UMAP-2:</b> %{y:.2f}<br>" +
                        "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive UMAP Visualization of Keystroke Clustering{title_suffix}<br><sub>Colored by True Labels</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='UMAP Dimension 1',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title='UMAP Dimension 2',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save as HTML
    filename = f"umap_clustering{title_suffix.lower().replace(' ', '_')}.html"
    output_path = os.path.join(img_dir, filename)
    fig.write_html(output_path, include_plotlyjs='cdn')
    
    print(f"Interactive UMAP visualization saved to: {output_path}")
    return embedding