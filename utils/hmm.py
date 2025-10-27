import os
import json
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import Levenshtein
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
from collections import Counter
import csv
import seaborn as sns


# Import configuration
from config import *

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)   


class BigramHMM:
    """Hidden Markov Model for keyboard acoustic eavesdropping using bigram language model"""
    
    def __init__(self, keys, n_states, n_clusters, 
                 max_iter, tol, bigram_path, img_dir, reporter=None):
        self.keys = keys
        self.n_states = n_states
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.bigram_path = bigram_path
        self.img_dir = img_dir
        self.reporter = reporter
        
        self.char_to_idx = {char: i for i, char in enumerate(keys)}
        self.transmat = None
        self.eta = None
        
    def calc_A(self):
        """Calculate transition matrix from bigram statistics - same as original"""
        if self.reporter:
            self.reporter.write("Phase 1: Loading bigram language model and calculating transition matrix")
        
        alphabet = list(self.keys)
        char_to_idx = {char: i for i, char in enumerate(alphabet)}
        n = len(alphabet)
        
        with open(self.bigram_path, "r", encoding="utf-8") as f:
            bigrams = json.load(f)
        
        if self.reporter:
            self.reporter.write(f"Loaded {len(bigrams)} bigram entries from char_bigrams.json")
        
        A_counts = np.zeros((n, n), dtype=np.float64)
        processed_bigrams = 0
        for bigram, count in bigrams.items():
            if len(bigram) != 2:
                continue
            a, b = bigram[0], bigram[1]
            if a in char_to_idx and b in char_to_idx:
                i, j = char_to_idx[a], char_to_idx[b]
                A_counts[i, j] += count
                processed_bigrams += 1
        
        if self.reporter:
            self.reporter.write(f"Processed {processed_bigrams} valid bigrams for transition matrix")
        
        row_sums = A_counts.sum(axis=1, keepdims=True)
        A_prob = A_counts / np.where(row_sums == 0, 1, row_sums)
        
        if self.reporter:
            self.reporter.write(f"Transition matrix calculated: {A_prob.shape}")
        
        return A_prob
    
    def init_emission_prob(self, c_spaces, boost_prob=0.95):
        """Initialize emission matrix - same as original"""
        if self.reporter:
            self.reporter.write("Phase 2: Initializing space-constrained emission matrix")
            self.reporter.write(f"Space clusters: {c_spaces}")
            self.reporter.write(f"Boost probability for space clusters: {boost_prob}")
        
        eta = np.full((self.n_states, self.n_clusters), 1.0 / self.n_clusters)
        eta[-1] = np.full(self.n_clusters, (1.0 - boost_prob) / (self.n_clusters - 1))
        for cs in c_spaces:
            eta[-1][cs] = boost_prob
        
        if self.reporter:
            self.reporter.write(f"Emission matrix initialized: {eta.shape}")
            self.reporter.write(f"Space state (index {self.n_states-1}) configured with {len(c_spaces)} space clusters")
        
        return eta
    
    def forward_backward(self, obs, transmat, eta):
        """Forward-backward algorithm - same as original"""
        n_states = transmat.shape[0]
        T = len(obs)
        
        alpha = torch.zeros((n_states, T), dtype=torch.float64)
        beta = torch.zeros((n_states, T), dtype=torch.float64)
        scale = torch.zeros(T, dtype=torch.float64)
        
        alpha[:, 0] = (1 / n_states) * eta[:, obs[0]]
        scale[0] = alpha[:, 0].sum()
        alpha[:, 0] /= scale[0]
        
        for t in range(1, T):
            alpha[:, t] = (transmat.T @ alpha[:, t - 1]) * eta[:, obs[t]]
            scale[t] = alpha[:, t].sum()
            alpha[:, t] /= scale[t]
        
        beta[:, -1] = 1 / scale[-1]
        for t in reversed(range(T - 1)):
            beta[:, t] = transmat @ (eta[:, obs[t + 1]] * beta[:, t + 1])
            beta[:, t] /= scale[t]
        
        return alpha, beta, scale
    
    def hmm_em(self, obs, transmat, eta_init):
        """EM algorithm - same as original"""
        if self.reporter:
            self.reporter.write("Phase 3: Starting EM algorithm for parameter estimation")
            self.reporter.write(f"Max iterations: {self.max_iter}, Tolerance: {self.tol}")
        
        n_states, n_clusters = eta_init.shape
        eta = eta_init.clone()
        prev_log_likelihood = -float('inf')
        
        for iteration in tqdm(range(self.max_iter)):
            alpha, beta, scale = self.forward_backward(obs, transmat, eta)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(dim=0, keepdim=True)
            
            eta_new = torch.zeros_like(eta)
            for k in range(n_clusters):
                eta_new[:, k] = gamma[:, obs == k].sum(dim=1)
            
            eta_new = eta_new / eta_new.sum(dim=1, keepdim=True)
            eta_new[-1] = eta_init[-1]  # Space state is fixed
            
            log_likelihood = torch.sum(torch.log(scale))
            
            # Log progress every 100 iterations
            if self.reporter and iteration % 100 == 0:
                self.reporter.write(f"EM iteration {iteration}: log_likelihood = {log_likelihood:.4f}")
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.reporter:
                    self.reporter.write(f"EM algorithm converged at iteration {iteration}")
                    self.reporter.write(f"Final log_likelihood: {log_likelihood:.4f}")
                break
            prev_log_likelihood = log_likelihood
            eta = eta_new
        
        if self.reporter:
            self.reporter.write("EM algorithm completed")
        
        return eta
    
    def viterbi_decode(self, obs, transmat, eta):
        """Viterbi algorithm - same as original"""
        if self.reporter:
            self.reporter.write("Phase 4: Starting Viterbi decoding for optimal sequence")
        
        n_states = transmat.shape[0]
        T = len(obs)
        
        if self.reporter:
            self.reporter.write(f"Decoding sequence of length {T} with {n_states} states")
        
        log_trans = torch.log(transmat + 1e-12)
        log_emit = torch.log(eta + 1e-12)
        
        delta = torch.full((n_states, T), float('-inf'), dtype=torch.float64)
        psi = torch.zeros((n_states, T), dtype=torch.int64)
        
        delta[:, 0] = torch.log(torch.full((n_states,), 1.0 / n_states)) + log_emit[:, obs[0]]
        
        for t in range(1, T):
            for j in range(n_states):
                trans_scores = delta[:, t - 1] + log_trans[:, j]
                best_prev_state = torch.argmax(trans_scores)
                delta[j, t] = trans_scores[best_prev_state] + log_emit[j, obs[t]]
                psi[j, t] = best_prev_state
        
        states = torch.zeros(T, dtype=torch.int64)
        states[-1] = torch.argmax(delta[:, -1])
        for t in reversed(range(1, T)):
            states[t - 1] = psi[states[t], t]
        
        if self.reporter:
            self.reporter.write("Viterbi decoding completed")
        
        return states
  
    def evaluate_text_acc(self, pred, orign):
        """Evaluate text accuracy - same as original"""
        distance = Levenshtein.distance(pred, orign)
        max_len = max(len(pred), len(orign))
        distance_acc = 1.0 - distance / max_len
        
        # Character-wise accuracy by direct comparison
        min_len = min(len(pred), len(orign))
        correct_chars = sum(1 for i in range(min_len) if pred[i] == orign[i])
        char_acc = correct_chars / max_len if max_len > 0 else 0.0
        
        return {"levenshtein": distance_acc, "char_wise": char_acc}
    
    def save_emission_heatmap(self, eta, title, filename):
        """Save emission matrix as heatmap"""
        # Ensure img directory exists
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Convert to numpy if tensor
        if torch.is_tensor(eta):
            eta_np = eta.detach().cpu().numpy()
        else:
            eta_np = eta
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(eta_np, 
                    xticklabels=[f'C{i}' for i in range(eta_np.shape[1])],
                    yticklabels=list(self.keys),
                    cmap='Reds',
                    cbar_kws={'label': 'Emission Probability'},
                    linewidth=.3
                    )
        
        plt.title(title)
        plt.xlabel('Cluster Index')
        plt.ylabel('Character State')
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.img_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.reporter:
            self.reporter.write(f"Saved emission matrix heatmap: {filepath}")
    
    def run_inference(self, clusters, c_spaces, true_text=None):
        """Run full inference pipeline with enhanced evaluation"""
        if self.reporter:
            self.reporter.write("Starting HMM inference pipeline")
            self.reporter.write(f"Input clusters: {len(clusters)} observations")
            self.reporter.write(f"Space clusters: {len(c_spaces)} clusters")
        
        transmat = self.calc_A()
        eta_init = self.init_emission_prob(c_spaces)
        
        nt = len(clusters)
        obs = torch.tensor(clusters[:nt], dtype=torch.int64)
        transmat = torch.tensor(transmat, dtype=torch.float64)
        eta = torch.tensor(eta_init, dtype=torch.float64)
        
        if self.reporter:
            self.reporter.write(f"Tensor conversion completed. Observation sequence length: {len(obs)}")
        
        # Save initial emission matrix heatmap
        self.save_emission_heatmap(eta, "Initial Emission Matrix (Before EM)", "emission_matrix_initial.png")
        
        eta_final = self.hmm_em(obs, transmat, eta)
        
        # Save final emission matrix heatmap
        self.save_emission_heatmap(eta_final, "Final Emission Matrix (After EM)", "emission_matrix_final.png")
        
        # Standard Viterbi decoding
        states = self.viterbi_decode(obs, transmat, eta_final)
        decoded_text = ''.join([self.keys[s] for s in states.tolist()])
        
        results = {'decoded_text': decoded_text}
        
        if self.reporter:
            self.reporter.write(f"Inference completed. Decoded text length: {len(decoded_text)} characters")
            self.reporter.write(f"Text preview: '{decoded_text[:50]}{'...' if len(decoded_text) > 50 else ''}'")
        
        return results

class FeedbackDataset:
    def __init__(self, hmm_output, gemini_output, reporter=None):
        self.hmm_output = hmm_output
        self.gemini_output = gemini_output
        self.word1 = hmm_output.split()
        self.word2 = gemini_output.split()
        self.reporter = reporter

    def get_similarity(self, s1, s2):
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def find_best_greedy_match(self, seg1, seg2, max_n=3):
        correspondences = []
        i, j = 0, 0
        while i < len(seg1) or j < len(seg2):
            if i >= len(seg1):
                correspondences.append(("", " ".join(seg2[j:])))
                break
            if j >= len(seg2):
                correspondences.append((" ".join(seg1[i:]), ""))
                break

            best_match = (0, 1, 1)
            for n1 in range(1, min(len(seg1) - i, max_n) + 1):
                for n2 in range(1, min(len(seg2) - j, max_n) + 1):
                    phrase1 = " ".join(seg1[i:i+n1])
                    phrase2 = " ".join(seg2[j:j+n2])
                    score = self.get_similarity(phrase1, phrase2)
                    if score > best_match[0] or (score == best_match[0] and (n1 + n2) < (best_match[1] + best_match[2])):
                        best_match = (score, n1, n2)
            
            len1, len2 = best_match[1], best_match[2]
            correspondences.append((" ".join(seg1[i:i+len1]), " ".join(seg2[j:j+len2])))
            i += len1
            j += len2
        return correspondences

    def get_word_correspondences(self):
        matcher = difflib.SequenceMatcher(None, self.word1, self.word2, autojunk=False)
        final_correspondences = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            seg1, seg2 = self.word1[i1:i2], self.word2[j1:j2]
            if tag == 'replace':
                final_correspondences.extend(self.find_best_greedy_match(seg1, seg2))
            else:
                final_correspondences.append((" ".join(seg1), " ".join(seg2)))
        return final_correspondences

    def make_feedback_label(self):
        correspondences = self.get_word_correspondences()
        filtered_pairs = []
        for original, corrected in correspondences:
            if original and corrected and len(original) == len(corrected) and self.get_similarity(original, corrected) >= FEEDBACK_TH:
                filtered_pairs.append((original, corrected))
        
        swapped_line = self.hmm_output
        for original, corrected in filtered_pairs:
            swapped_line = swapped_line.replace(original, corrected)
            
        mask = [0] * len(self.hmm_output)
        for original, _ in filtered_pairs:
            start_index = 0
            while (pos := self.hmm_output.find(original, start_index)) != -1:
                for i in range(pos, pos + len(original)):
                    mask[i] = 1
                start_index = pos + 1
        
        return {"origin": self.hmm_output, "gemini": self.gemini_output, "swapped": swapped_line, "mask": mask}

    def create_feedback_dataset(self):
        if self.reporter:
            self.reporter.write("Creating feedback dataset, including space keys in training/inference.")

        # Check for required files
        for f in [COMPRESSED_FEATURE_FILE, SPACE_ESTIMATION_FILE]:
            if not os.path.exists(f):
                message = f"Required file for feedback dataset not found at {f}"
                if self.reporter: self.reporter.write(f"ERROR: {message}")
                raise FileNotFoundError(message)

        # Load all necessary data
        features = np.load(COMPRESSED_FEATURE_FILE)
        space_predictions = np.load(SPACE_ESTIMATION_FILE)
        labels_data = self.make_feedback_label()
        confidence_mask = np.array(labels_data["mask"])
        swapped_text = labels_data["swapped"]

        # Validate data integrity
        if not (len(features) == len(swapped_text) == len(space_predictions) == len(confidence_mask)):
            message = f"Data length mismatch: Features({len(features)}), Text({len(swapped_text)}), Spaces({len(space_predictions)})"
            if self.reporter: self.reporter.write(f"ERROR: {message}")
            raise ValueError(message)

        # --- Modified Logic to Include Spaces ---
        # 1. Identify high-confidence keys (including spaces)
        is_high_confidence_mask = (confidence_mask == 1)

        # 2. Determine final indices for training and inference (all characters including spaces)
        # Train: Must be high-confidence (regardless of space/non-space)
        train_indices = np.where(is_high_confidence_mask)[0]
        # Inference: Must be low-confidence (regardless of space/non-space)
        inference_indices = np.where(~is_high_confidence_mask)[0]

        # Create datasets based on the new indices
        train_features = features[train_indices]
        train_labels = np.array(list(swapped_text))[train_indices]
        
        inference_features = features[inference_indices]
        
        dataset = {
            "train": {"features": train_features, "labels": train_labels},
            "inference": {"features": inference_features, "indices": inference_indices},
            "swapped_text": swapped_text
        }

        if self.reporter:
            self.reporter.write("Feedback dataset created successfully (spaces included).")
            self.reporter.write(f"  - Total samples: {len(features)}")
            self.reporter.write(f"  - Training samples (high-confidence): {len(train_features)}")
            self.reporter.write(f"  - Inference samples (low-confidence): {len(inference_features)}")

            # Log training label statistics
            if len(train_labels) > 0:
                label_counts = Counter(train_labels)
                sorted_counts = sorted(label_counts.items())
                self.reporter.write("Training Label Distribution (including spaces):")
                for char, count in sorted_counts:
                    if char == ' ':
                        self.reporter.write(f"  - 'space': {count}")
                    else:
                        self.reporter.write(f"  - '{char}': {count}")
        
        return dataset

class FeedbackClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=128, dropout_rate=0.4):
        super(FeedbackClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(latent_dim // 2, num_classes)
        )
        self.char_to_idx = {char: i for i, char in enumerate(KEYS)}
        self.idx_to_char = {i: char for i, char in enumerate(KEYS)}

    def forward(self, x):
        return self.net(x)

    def train_model(self, train_features, train_labels, epochs=500, batch_size=16, learning_rate=1e-4, reporter=None):
        if reporter: reporter.write("Starting feedback model training.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
        label_indices = [self.char_to_idx.get(label, -1) for label in train_labels]
        labels_tensor = torch.tensor(label_indices, dtype=torch.long).to(device)

        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 and reporter:
                reporter.write(f"Feedback Model Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

        if reporter: reporter.write("Feedback model training completed.")

    def predict_model(self, inference_features, reporter=None):
        if reporter: reporter.write("Starting feedback model inference.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        features_tensor = torch.tensor(inference_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = self(features_tensor)
            _, predicted_indices = torch.max(outputs, 1)
        
        predicted_chars = [self.idx_to_char[idx.item()] for idx in predicted_indices]
        
        if reporter: reporter.write(f"Inference completed. Predicted {len(predicted_chars)} characters.")
        return predicted_chars

def run_feedback_cycle(feedback_dataset, reporter=None):
    if reporter: reporter.write("Running feedback learning cycle.")

    train_data = feedback_dataset["train"]
    inference_data = feedback_dataset["inference"]
    swapped_text = feedback_dataset["swapped_text"]

    if len(train_data["features"]) == 0:
        if reporter: reporter.write("No training data available for feedback model. Skipping training.")
        return swapped_text

    # Initialize and train the model
    model = FeedbackClassifier(input_dim=LATENT_DIM, num_classes=N_STATES)
    model.train_model(train_data["features"], train_data["labels"], reporter=reporter)

    # Perform inference
    if len(inference_data["features"]) > 0:
        predicted_chars = model.predict_model(inference_data["features"], reporter=reporter)
        
        # Integrate predictions back into the text
        final_text_list = list(swapped_text)
        for i, char_index in enumerate(inference_data["indices"]):
            final_text_list[char_index] = predicted_chars[i]
        
        final_text = "".join(final_text_list)
        if reporter: reporter.write("Feedback predictions integrated into final text.")
    else:
        final_text = swapped_text
        if reporter: reporter.write("No inference data. Returning swapped text as final.")

    return final_text

def correct_text(text_to_correct, reporter=None):
    """
    Corrects typos in a given text using the Gemini API.
    """
    if reporter:
        reporter.write("Phase 5: Starting Gemini text correction")
        reporter.write(f"Input text length: {len(text_to_correct)} characters")
    
    try:
        # Load environment variables and configure the API key
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            if reporter:
                reporter.write("ERROR: GOOGLE_API_KEY not found in .env file or environment")
            raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
        
        if reporter:
            reporter.write("Gemini API key loaded successfully")
        
        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        #model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        
        
        if reporter:
            reporter.write("Gemini model initialized ")

        # Create the prompt
        prompt = (
        # 役割を付与
        "You are an expert cryptanalyst tasked with correcting garbled text recovered from an acoustic "
        "side-channel attack on a keyboard. The text contains frequent character-level errors "
        "(substitutions, deletions, insertions), but the underlying content is standard English.\n\n"
        
        "Your goal is to reconstruct the original text. Follow these instructions:\n"
        "1. Correct character errors to form valid and contextually appropriate English words.\n"
        # スペースキーに関する制約を明確化
        "2. Preserve spaces: The positions of space characters are highly reliable. "
        "Do not add or remove them unless absolutely necessary to correct a word boundary.\n"
        "3. Output format: Return only the clean, corrected text without any preamble.\n\n"
        
        "Now, correct the following text:\n"
        f"Original text: '{text_to_correct}'\n\n"
        "Corrected text:"
    )

        if reporter:
            reporter.write("Sending text correction request to Gemini API")

        # Generate content and return the corrected text
        response = model.generate_content(prompt)
        corrected_text = re_text(response.text)
        
        if reporter:
            reporter.write("Text correction completed successfully")
            reporter.write(f"Corrected text length: {len(corrected_text)} characters")
            reporter.write(f"Corrected text preview: '{corrected_text[:50]}{'...' if len(corrected_text) > 50 else ''}'")

        if len(corrected_text) > N_KEYSTROKES+100:
            return corrected_text[:N_KEYSTROKES+100]
        else:
            return corrected_text
        
    except Exception as e:
        if reporter:
            reporter.write(f"ERROR: Gemini text correction failed: {str(e)}")
        print(f"Warning: Gemini correction failed: {e}")
        return text_to_correct  # Return original text if correction fails

def re_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

