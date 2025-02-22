# config.py
import os

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Paths and Directories
# -------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # Folder for input car images
RESULTS_DIR = os.path.join(BASE_DIR, "results")          # Folder to store results and model files
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "car_model_classifier.h5")  # Path to save the trained model

# Create the results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# -------------------------
# Image and Model Settings
# -------------------------
IMAGE_SIZE = (224, 224)  # Input image dimensions (width, height)
NUM_CHANNELS = 3         # Number of image channels (RGB)

# Choose your model architecture: e.g., "ResNet50", "VGG16", etc.
MODEL_NAME = "ResNet50"

# -------------------------
# Training Configuration
# -------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Whether to use data augmentation during training
AUGMENTATION = True

# Random seed for reproducibility
SEED = 42
