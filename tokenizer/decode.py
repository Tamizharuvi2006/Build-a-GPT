from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")

def decode(token_ids):
    """
    Convert token IDs back to text.
    """
    return tokenizer.decode(token_ids)


if __name__ == "__main__":

    ids = [2, 145, 76, 88]

    print(decode(ids))