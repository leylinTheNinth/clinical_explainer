from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    AutoTokenizer
)

class ModelType(Enum):
    """Enum for different types of models"""
    ENCODER = auto()  # BERT-style models
    DECODER = auto()  # GPT/Mistral-style models

class ProcessingStrategy(ABC):
    """Base class for all model processing strategies"""
    
    def __init__(self, model, tokenizer, explainer_types: Optional[List[str]] = None):
        """
        Initialize strategy with specified explainer types
        Args:
            model: The loaded model
            tokenizer: The model's tokenizer
            explainer_types: List of explainer types to use ('lime', 'shap', 'tokenShap')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.explainer_types = explainer_types or []  # Default to empty list if None
        self.explainers = self._setup_explainers()  # Setup only requested explainers
        
    def _setup_explainers(self) -> Dict:
        """Setup only the requested explainer types"""
        explainers = {}
        
    @abstractmethod
    def preprocess(self, case: Dict) -> Dict:
        """Preprocess a single case for the model"""
        pass
        
    @abstractmethod
    def predict(self, processed_case: Dict) -> Dict:
        """Generate prediction for processed case"""
        pass
    
    def explain(self, processed_case: Dict, prediction: Dict) -> Dict:
        """Run all explainers and return results"""
        explanations = {}
        for name, explainer in self.explainers.items():
            try:
                exp = explainer.explain(processed_case, prediction)
                explanations[name] = exp
            except Exception as e:
                print(f"Error in {name} explainer: {str(e)}")
        return explanations

class StrategyFactory:
    """Factory for creating appropriate processing strategy"""
    
    @staticmethod
    def create_strategy(
        model_type: ModelType,
        model_name: str,
        explainer_types: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> ProcessingStrategy:
        """
        Create strategy with model loading and optional quantization
        
        Args:
            model_type: Type of model (ENCODER/DECODER)
            model_name: HuggingFace model name/path
            device: Device to load model on
            **kwargs: Additional args like quantize=True/False
            
        Returns:
            Configured ProcessingStrategy instance
        """
        try:
            if model_type == ModelType.DECODER:
                raise RuntimeError(f"Decoder strategy is not implemented.")
                '''
                # Load decoder (GPT-style) model with optional quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=kwargs.get('quantize', True),
                    torch_dtype=torch.float16
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Import here to avoid circular imports
                from .decoder_strategy import DecoderStrategy
                return DecoderStrategy(model, tokenizer, explainer_types)
                '''
            elif model_type == ModelType.ENCODER:
                # Load encoder (BERT-style) model
                model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Import here to avoid circular imports
                from .encoder_strategy import EncoderStrategy
                return EncoderStrategy(model, tokenizer, explainer_types)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to create strategy: {str(e)}")