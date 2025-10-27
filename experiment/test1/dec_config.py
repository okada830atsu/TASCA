
import os

# =============================================
# Directory and File Paths
# =============================================

# Base directories
SCRIPT_DIR              = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT            = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))


# Data directories
DATA_DIR        = "./data/direct/segmented"

# Output directories
LOG_FILE                = "./result/log/test1.log"
IMG_DIR                 = "./result/img"

# Feature files (Evaluate mode)
AE_FEATURE_FILE         = "./data/feature/ae_features.npy"
AE_LABEL_FILE           = "./data/feature/ae_labels.npy"


# Clustering output files
CLUSTER_ESTIMATION_FILE = "./feature/cluster_estimation.npy"
SPACE_ESTIMATION_FILE   = "./feature/space_estimation.npy"
SPACE_CLUSTERS_FILE     = "./feature/space_clusters.npy"

# HMM input files
BIGRAM_FILE             = "./data/language/char_bigrams.json"
ANSWER_FILE             = "./data/language/answer.txt"

# Model files
AE_MODEL_FILE           = "./data/model/autoencoder_model.pth"

# =============================================
# Number of Keystrokes
# =============================================
# [204, 453, 611, 786, 926, 1193, 1492, 1690] for solar-system article
N_KEYSTROKES            = 926


# =============================================
# Audio Processing Parameters
# =============================================

# Audio settings
SAMPLE_RATE             = 44100
N_MFCC                  = 128
N_MELS                  = 64
N_FFT                   = 1024
HOP_LENGTH              = 512

# Character settings
KEYS                    = "abcdefghijklmnopqrstuvwxyz "
N_STATES                = 27            # Same as N_STATES for HMM

# =============================================
# Deep Learning Parameters
# =============================================

# Autoencoder architecture
LATENT_DIM              = 128            # Latent space dimension

# Training parameters - Autoencoder
AE_EPOCHS               = 2000
AE_LEARNING_RATE        = 2e-3
BATCH_SIZE              = 32

# Training parameters - DEC
N_CLUSTERS              = 30            # Number of clusters for overclustering
DEC_EPOCHS              = 2000
DEC_LEARNING_RATE       = 1e-3
UPDATE_INTERVAL         = 100           # Evaluation interval (epochs)
TOLERANCE               = 5e-5          # Convergence tolerance

# =============================================
# Space Detection Parameters
# =============================================

# Semi-supervised space detection
#SPACE_SAMPLE            = [8, 13, 21, 29, 34, 42, 45, 47]#, 59, 66, 69, 73, 79, 82, 94, 103]   # Known space sample indices
SPACE_SAMPLE            = [4,7,13,27, 30, 36, 39, 45, 52, 56, 64, 72]#, 372, 377, 386]
SPACE_DETECT_RADIUS     = 1            # Detection radius in UMAP space

# =============================================
# HMM Parameters
# =============================================

# HMM training parameters
MAX_ITER                = 1000          # Maximum EM iterations
TOL                     = 5e-4          # EM convergence tolerance
BOOST_PROB              = 0.95          # Probability boost for space clusters

# =============================================
# Feedback Learning Parameters
# =============================================

FEEDBACK_ITERATIONS = 5
FEEDBACK_TH = 0.6
USE_LINEAR_FEEDBACK_CLASSIFIER = True  # True: Linear classifier, False: Neural Network

# =============================================
# Visualization Parameters
# =============================================

# UMAP parameters
UMAP_N_NEIGHBORS        = 15
UMAP_MIN_DIST           = 0.1
UMAP_N_COMPONENTS       = 2
UMAP_RANDOM_STATE       = 42
UMAP_METRIC             = 'cosine'

# Plot settings
FIGURE_WIDTH            = 1200
FIGURE_HEIGHT           = 800
HEATMAP_WIDTH           = 12
HEATMAP_HEIGHT          = 8



