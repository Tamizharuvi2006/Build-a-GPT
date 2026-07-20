from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
TOKENIZER = ROOT / "data" / "tokenizer"

CHECKPOINTS = ROOT / "checkpoints"

LOGS = ROOT / "logs"