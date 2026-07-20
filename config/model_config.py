"""
Model Configuration
"""

# Vocabulary
VOCAB_SIZE = 8000

# Sequence
CONTEXT_LENGTH = 128

# Embedding
EMBED_DIM = 192

# Transformer
NUM_LAYERS = 6

# Feed Forward
FFN_MULTIPLIER = 4

# Dropout
DROPOUT = 0.1

# Attention
HEAD_DIM = 32
NUM_QUERY_HEADS = 6
NUM_KV_HEADS = 2
ATTENTION_DROPOUT = 0.1

USE_GQA = True
USE_SLIDING_WINDOW = True
WINDOW_SIZE = 64

# Positional Encoding
USE_ROPE = True

# Normalization
USE_RMSNORM = True

# Activation
ACTIVATION = "swiglu"

# Output
BIAS = False

# Advanced Features
USE_MOE = True
NUM_EXPERTS = 4
TOP_K_EXPERTS = 2

USE_MEMORY = True
MEMORY_SIZE = 128

USE_REASONING = True
REASONING_ITERATIONS = 3

USE_PLANNER = True
PLAN_DIM = 192

USE_LONG_CONTEXT = True
LONG_CONTEXT_SCALE = 4.0

USE_FUSION = True
CONTEXT_DIM = 192