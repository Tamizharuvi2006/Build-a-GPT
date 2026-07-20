import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

# ----------------------------
# Paths
# ----------------------------

DATA_PATH = Path("data/raw/tinystories_100mb.txt")

TOKENIZER_PATH = Path("data/tokenizer/tokenizer.json")

OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load tokenizer
# ----------------------------

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# ----------------------------
# Read dataset
# ----------------------------

print("Reading dataset...")

text = DATA_PATH.read_text(encoding="utf-8")

print(f"Characters : {len(text):,}")

# ----------------------------
# Encode
# ----------------------------

print("Encoding...")

tokens = tokenizer.encode(text).ids

print(f"Tokens : {len(tokens):,}")

# ----------------------------
# Split
# ----------------------------

split = int(len(tokens) * 0.9)

train_ids = tokens[:split]
val_ids = tokens[split:]

print(f"Train Tokens : {len(train_ids):,}")
print(f"Validation Tokens : {len(val_ids):,}")

# ----------------------------
# Save
# ----------------------------

np.array(train_ids, dtype=np.uint32).tofile(
    OUTPUT_DIR / "train.bin"
)

np.array(val_ids, dtype=np.uint32).tofile(
    OUTPUT_DIR / "val.bin"
)

print("\nDone!")