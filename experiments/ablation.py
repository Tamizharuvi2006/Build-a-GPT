"""Framework for running ablation studies on model architectures."""
from typing import Dict, Callable, Any

class AblationStudy:
    """Manages and compares multiple experimental configurations."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = {}
        
    def run_experiment(self, name: str, config_override: Dict[str, Any], 
                       train_fn: Callable, eval_fn: Callable) -> float:
        """Run a single ablation experiment."""
        config = self.base_config.copy()
        config.update(config_override)
        
        print(f"Running experiment: {name}")
        model = train_fn(config)
        score = eval_fn(model)
        
        self.results[name] = score
        return score
        
    def compare_results(self) -> str:
        """Generate a summary table of all experiments."""
        lines = ["Ablation Study Results", "----------------------"]
        for name, score in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{name:<20}: {score:.4f}")
        return "\n".join(lines)
