from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable
from enum import Enum, auto

class ExplainerType(Enum):
    """Enum for different types of explainers"""
    LIME = auto()
    SHAP = auto()

class BaseExplainer(ABC):
    @abstractmethod
    def create_explainer_function(self, options: List[str], tokenizer: Any, model: Any, device: str) -> Callable:
        """Creates prediction function with necessary context"""
        pass
    
    @abstractmethod
    def explain(self, question: str, predict_fn: Callable, class_names: List[str], target_label: int):
        """Generates explanation using the prediction function"""
        pass

__all__ = ['BaseExplainer', 'ExplainerType']
