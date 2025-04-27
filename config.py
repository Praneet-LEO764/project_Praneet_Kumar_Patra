"""
Configuration file for the unicorn image classification model.
Contains all hyperparameters and constants used throughout the project.
"""

# Training hyperparameters
BATCH_SIZE = 64  # Increased from 50 for better GPU utilization
EPOCHS = 20  # Increased from 10 for better convergence
LEARNING_RATE = 0.001  # Added learning rate parameter
WEIGHT_DECAY = 1e-5  # Added weight decay for regularization

# Data processing parameters
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
INPUT_CHANNELS = 3
NUM_WORKERS = 4  # Added for faster data loading
VALIDATION_SPLIT = 0.2  # Added for train/validation split

# Model parameters
NUM_CLASSES = 2  # Binary classification

# Device configuration
USE_CUDA = True  # Flag to use GPU if available
