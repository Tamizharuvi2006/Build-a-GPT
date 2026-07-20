from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

DATA_FILE = Path("data/raw/tinystories_100mb.txt")
SAVE_DIR = Path("data/tokenizer")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

from config.tokenizer_config import VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=[f"[{k}]" for k in SPECIAL_TOKENS.keys()],
)

print("Training tokenizer...")

tokenizer.train([str(DATA_FILE)], trainer)

tokenizer.save(str(SAVE_DIR / "tokenizer.json"))

print("Tokenizer saved!")