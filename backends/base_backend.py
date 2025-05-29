from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio


class BaseBackend(ABC):
    """Abstract base class for all inference backends."""
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load model, tokenizer, etc.)."""
        pass
    
    @abstractmethod
    def generate(self, tokenized_prompts: List[List[int]], prompt_strings: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a list of pre-tokenized prompts synchronously.
        
        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs)
            prompt_strings: Optional list of prompt strings (for backends that need strings)
            
        Returns:
            List of dictionaries with standardized output format
        """
        pass
    
    @abstractmethod
    def generate_async(self, tokenized_prompts: List[List[int]], prompt_strings: Optional[List[str]] = None, **kwargs) -> List[asyncio.Future]:
        """
        Generate responses for a list of pre-tokenized prompts asynchronously.
        
        This method returns immediately with a list of futures that will resolve
        to the generation results.
        
        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs)
            prompt_strings: Optional list of prompt strings (for backends that need strings)
            
        Returns:
            List of futures that will resolve to dictionaries with standardized output format
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown() 