"""
Configuration for inference.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"
USE_KV_CACHE = True
STREAM_OUTPUT = True
