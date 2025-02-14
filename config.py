import torch

# Paths
DATA_PATH = "data/imdb_reviews.csv"
MODEL_PATH = "models/sentiment_model.pth"

# Training settings
BATCH_SIZE = 64
N_EPOCHS = 5
LEARNING_RATE = 0.0005

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
OUTPUT_DIM = 1
DROPOUT = 0.5

# Device selection
def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) on macOS.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda")
    else:
        print("Using CPU (No compatible GPU found).")
        return torch.device("cpu")

DEVICE = get_device()