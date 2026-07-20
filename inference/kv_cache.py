class InferenceKVCache:
    """
    Stores past key-values per layer for efficient autoregressive generation.
    """
    def __init__(self):
        self.cache = {}

    def update(self, layer_idx, new_key, new_value):
        """
        Updates the cache for a specific layer.
        """
        self.cache[layer_idx] = (new_key, new_value)

    def get(self, layer_idx):
        """
        Retrieves the cached key-values for a layer.
        Returns None if the layer is not in the cache.
        """
        return self.cache.get(layer_idx, None)

    def reset(self):
        """
        Clears the cache for a new generation pass.
        """
        self.cache.clear()
