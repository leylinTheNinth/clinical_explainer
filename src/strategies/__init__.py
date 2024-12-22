from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

class ModelType(Enum):
    """Enum for different types of models"""
    ENCODER = auto()  # BERT-style models
    DECODER = auto()  # GPT-style models

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
        return explainers
        
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


from .encoder_strategy import EncoderStrategy
from .decoder_strategy import DecoderStrategy

class StrategyFactory:
    """Factory for creating appropriate processing strategy"""
    
    @staticmethod
    def create_strategy(
        model_type: ModelType,
        model_name: str,
        explainer_types: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sampling_ratio: float = 0.0,  
        max_length: int = 256,
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
                # Check if model needs quantization
                should_quantize = StrategyFactory._should_quantize(model_name)
                if should_quantize:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    # For quantized models, let device_map handle placement
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    print("\n8bit quantization applied.\n")
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    ).to(device)
                    
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                return DecoderStrategy(model, 
                                       tokenizer, 
                                       explainer_types, 
                                       max_length=max_length,
                                       sampling_ratio=sampling_ratio)
            
            elif model_type == ModelType.ENCODER:
                # Load encoder (BERT-style) model
                model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                return EncoderStrategy(model, tokenizer, explainer_types)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to create strategy: {str(e)}")
    
    @staticmethod
    def _should_quantize(model_name: str) -> bool:
        """Determine if model needs quantization based on size/type"""
        # Models that typically need quantization
        large_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b",
            "mistralai/Mixtral-8x7B",
        ]
        
        # Check if model is in large models list
        return any(model_name.startswith(prefix) for prefix in large_models)