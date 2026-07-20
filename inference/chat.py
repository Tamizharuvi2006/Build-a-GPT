class ChatSession:
    """
    Manages multi-turn conversations.
    """
    def __init__(self, model, tokenizer_encode, tokenizer_decode, system_prompt=None):
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.history = []
        if system_prompt:
            self.history.append(f"[BOS] system\n {system_prompt}\n")

    def chat(self, user_message, **generate_kwargs):
        """
        Processes a user message and returns the assistant's response.
        """
        from .generate import generate_text
        
        # Append user message
        self.history.append(f"user\n {user_message}\n assistant\n")
        
        # Create full prompt
        prompt = "".join(self.history)
        
        # Generate full text
        response_text = generate_text(
            self.model, 
            prompt, 
            self.encode, 
            self.decode, 
            **generate_kwargs
        )
        
        # Extract just the new response part
        new_text = response_text[len(prompt):]
        
        # Save assistant response to history
        self.history[-1] = self.history[-1] + new_text + "\n"
        
        return new_text
