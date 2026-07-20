"""
Training Configuration
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16



LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

BETAS = (0.9, 0.95)

GRAD_CLIP = 1.0

SAVE_EVERY = 1

MAX_TRAIN_BATCHES = 500
MAX_VAL_BATCHES = 50
EPOCHS = 5
SEED = 42

CHECKPOINT_DIR = "checkpoints"