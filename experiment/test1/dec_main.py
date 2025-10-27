
import sys
sys.path.append("../..")

from utils.feature import FeatureExtractor
from utils.clustering import TraditionalKMeans, DECNet, DEC, evaluate_clustering, load_data
#from utils.hmm import BigramHMM
from utils.reporter import Reporter
from experiment.test1.dec_config import *

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
from umap import UMAP
import argparse
import datetime


def main():

    # =============================================
    # 初期設定
    # =============================================

    reporter = Reporter(LOG_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =============================================
    # データ読み込み
    # =============================================

    data, labels = load_data(DATA_DIR, N_KEYSTROKES, SAMPLE_RATE, label=True)
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    print(f"Number of classes: {len(np.unique(label_ids))}")

    # =============================================
    # 特徴量抽出
    # =============================================
    feature_extractor = FeatureExtractor()
    features = feature_extractor.complex_mfcc(data, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH)
    print(features.shape)

    # =============================================
    # クラスタリング
    # =============================================

    model = DECNet(features.shape[1], LATENT_DIM, N_CLUSTERS).to(device)
    trainer = DEC(N_CLUSTERS, model, LATENT_DIM, device, AE_MODEL_FILE, IMG_DIR, reporter)
    trainer.train_autoencoder(features, AE_EPOCHS, BATCH_SIZE, AE_LEARNING_RATE)
    
    print(f"\nInitializing cluster centers (N_CLUSTERS={N_CLUSTERS})...")
    features_tensor = torch.from_numpy(features).to(device)
    trainer.model.eval()
    
    with torch.no_grad():
        encoded_features, _, _ = trainer.model(features_tensor)
    
    # K-means with multiple initializations
    best_kmeans = None
    best_inertia = float('inf')
    
    for i in range(5):
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=42+i, max_iter=500)
        kmeans.fit(encoded_features.cpu().numpy())
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans
    
    trainer.model.clustering_layer.data = torch.tensor(best_kmeans.cluster_centers_).to(device)
    print("Cluster initialization completed.")
    
    # Train DEC without accuracy evaluation
    trainer.train_dec(features, DEC_EPOCHS, DEC_LEARNING_RATE, UPDATE_INTERVAL, TOLERANCE)
    trainer.model.eval()  # Ensure dropout/inference layers switch to eval mode before clustering
    with torch.no_grad():
        _, _, q = trainer.model(features_tensor)
    final_predictions = q.argmax(1).cpu().numpy()

    clustering_results = evaluate_clustering(final_predictions, labels, le, IMG_DIR, "dec", reporter)


    return

    # ============================================= #
    # Setting and Loading
    # ============================================= #
     
    # Load input data
    clusters = np.load(CLUSTER_ESTIMATION_FILE)
    c_spaces = np.load(SPACE_CLUSTERS_FILE)
    
    reporter.write(f"Loaded cluster data: {len(clusters)} observations")
    reporter.write(f"Loaded space clusters: {c_spaces}")
    
    # Load ground truth
    with open(ANSWER_FILE, 'r') as f:
        true_text = f.read().strip()[:N_KEYSTROKES]
    
    reporter.write(f"Loaded ground truth text: {len(true_text)} characters")
    reporter.write(f"Ground truth text: '{true_text}'")
    
    
    # ============================================= #
    # Initial HMM prediction
    # ============================================= #
    
    # Create HMM instance and run inference with enhanced evaluation
    hmm = BigramHMM(reporter=reporter)
    inference_results = hmm.run_inference(clusters, c_spaces, true_text)
    decoded_text = inference_results['decoded_text']
    
    # Evaluate original HMM results
    print("\n--- Evaluation (before correction) ---")
    performance = hmm.evaluate_text_acc(decoded_text, true_text)
    print(f"Levenshtein distance score: {performance['levenshtein']*100:.2f} %")
    
    reporter.write("Original HMM Results:")
    reporter.write(f"Levenshtein accuracy: {performance['levenshtein']*100:.2f}%")
    reporter.write(f"Original decoded text: '{decoded_text}'")

    return
    # Apply Gemini correction
    gemini_corrected_text = correct_text(decoded_text, reporter=reporter)
    
    print("\n--- Evaluation (after Gemini correction) ---")
    gemini_performance = hmm.evaluate_text_acc(gemini_corrected_text, true_text)
    print(f"Levenshtein distance score: {gemini_performance['levenshtein']*100:.2f} %")
    
    reporter.write("Gemini Corrected Results:")
    reporter.write(f"Levenshtein accuracy: {gemini_performance['levenshtein']*100:.2f}%")
    reporter.write(f"Gemini corrected text: '{gemini_corrected_text}'")
    
    scores = []
    scores.append(performance['levenshtein']*100)
    scores.append(gemini_performance['levenshtein']*100)
    
    
    # ============================================= #
    # Iterative Feedback Learning Cycle
    # ============================================= #
    print("\n" + "="*60)
    print("Starting Iterative Feedback Learning Cycle...")
    print("="*60)

    # Initial inputs for the first iteration
    current_hmm_output    = decoded_text
    current_gemini_output = gemini_corrected_text

    for i in range(FEEDBACK_ITERATIONS):
        iteration = i + 1
        print("\n" + "="*40)
        print(f"Feedback Iteration {iteration}/{FEEDBACK_ITERATIONS}")
        print("="*40)
        reporter.write(f"--- Starting Feedback Iteration {iteration}/{FEEDBACK_ITERATIONS} ---")

        # 1. Create feedback dataset
        feedback_builder = FeedbackDataset(current_hmm_output, current_gemini_output, reporter=reporter)
        try:
            feedback_dataset = feedback_builder.create_feedback_dataset()
            train_count = len(feedback_dataset["train"]["labels"])
            inference_count = len(feedback_dataset["inference"]["indices"])
            print(f"Feedback dataset created: {train_count} training samples, {inference_count} inference samples.")

            # 2. Run the feedback cycle (train and predict)
            feedback_corrected_text = run_feedback_cycle(feedback_dataset, reporter=reporter)

            # 3. Evaluate the result from the feedback model
            print(f"\n--- Evaluation (after Feedback Cycle - Iteration {iteration}) ---")
            feedback_performance = hmm.evaluate_text_acc(feedback_corrected_text, true_text)
            print(f"Levenshtein distance score: {feedback_performance['levenshtein']*100:.2f} %")
            reporter.write(f"Iteration {iteration} - Feedback Cycle Results:")
            reporter.write(f"  Levenshtein: {feedback_performance['levenshtein']*100:.2f}%, Char-wise: {feedback_performance['char_wise']*100:.2f}%")
            reporter.write(f"  Text after feedback: '{feedback_corrected_text}'")

            # 4. Apply final Gemini correction for this iteration
            print(f"\n--- Final Gemini Correction (Iteration {iteration}) ---")
            final_corrected_text = correct_text(feedback_corrected_text, reporter=reporter)

            # 5. Evaluate the final result for this iteration
            final_performance = hmm.evaluate_text_acc(final_corrected_text, true_text)
            print(f"Levenshtein distance score: {final_performance['levenshtein']*100:.2f} %")
            reporter.write(f"Iteration {iteration} - Final Gemini Corrected Results:")
            reporter.write(f"  Levenshtein: {final_performance['levenshtein']*100:.2f}%, Char-wise: {final_performance['char_wise']*100:.2f}%")
            reporter.write(f"  Final corrected text: '{final_corrected_text}'")
            
            scores.append(feedback_performance['levenshtein']*100)
            scores.append(final_performance['levenshtein']*100)
    
            # 6. Prepare inputs for the next iteration
            current_hmm_output    = feedback_corrected_text
            current_gemini_output = final_corrected_text

        except (FileNotFoundError, ValueError) as e:
            print(f"Error during feedback iteration {iteration}: {e}")
            reporter.write(f"ERROR: Feedback iteration {iteration} failed: {e}")
            break  # Exit the loop if an error occurs
    
    print("\nPerformance history:")
    for s in scores:
        print(f"{s:.2f}, ", end="")
    reporter.write(str(scores))
    
    print(f"\nResults logged to: {hmm_log_file}")
    reporter.write("HMM Text Reconstruction Experiment Completed")

    

if __name__ == "__main__":
    main()


