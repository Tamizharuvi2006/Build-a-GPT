"""Memory module for keeping track of conversation history."""
from typing import List, Dict

class ConversationMemory:
    """Stores conversation turns and formats them for the LLM."""
    
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        
    def add_turn(self, role: str, content: str):
        """Add a turn to the conversation history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
            
    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history."""
        return self.history
        
    def get_formatted_history(self) -> str:
        """Return history formatted as a single string."""
        formatted = []
        for turn in self.history:
            formatted.append(f"{turn['role']}: {turn['content']}")
        return "\n".join(formatted)
        
    def clear(self):
        """Clear the conversation history."""
        self.history = []
