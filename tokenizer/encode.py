from tokenizers import Tokenizer
from pathlib import Path

# Load tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")

def encode(text: str):
    """
    Convert text into token IDs.
    """
    return tokenizer.encode(text).ids


if __name__ == "__main__":

    sample = "Once upon a time there was a dragon."

    ids = encode(sample)

    print("Text:")
    print(sample)

    print("\nToken IDs:")
    print(ids)