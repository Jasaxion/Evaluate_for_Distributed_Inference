from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch

class BaseFramework(ABC):
    """Base class for all inference frameworks."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text from prompt and return metrics.
        
        Returns:
            Dict containing:
            - 'output': generated text
            - 'ttft': time to first token
            - 'tpot': time per output token
            - 'memory_used': peak memory usage
        """
        pass

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()