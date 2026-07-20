import sys

class TokenStreamer:
    """
    Streams tokens to stdout in real-time.
    """
    def __init__(self, tokenizer_decode_fn):
        self.decode = tokenizer_decode_fn

    def put(self, token_id):
        """
        Decodes and prints incrementally.
        """
        # Note: robust streaming might require handling subword boundaries better.
        text = self.decode([token_id])
        sys.stdout.write(text)
        sys.stdout.flush()

    def end(self):
        """
        Flushes the stream with a newline.
        """
        sys.stdout.write("\n")
        sys.stdout.flush()

class CallbackStreamer:
    """
    Streams tokens to a callback function.
    """
    def __init__(self, tokenizer_decode_fn, callback_fn):
        self.decode = tokenizer_decode_fn
        self.callback_fn = callback_fn

    def put(self, token_id):
        text = self.decode([token_id])
        self.callback_fn(text)
        
    def end(self):
        pass
