from typing import Dict, Any, Callable
from token_shap import TokenSHAP, StringSplitter
from . import BaseExplainer
import torch

class TokenSHAPExplainer(BaseExplainer):
    """TokenSHAP-based explanation generator for decoder models"""
    
    def __init__(self, 
                 sampling_ratio: float = 0.0,
                 batch_size: int = 1,
                 max_length: int = 256):
        super().__init__()
        self.sampling_ratio = sampling_ratio
        self.batch_size = batch_size
        self.max_length = max_length

    def create_explainer_function(self, model, tokenizer):
        # Create the model wrapper with memory management
        wrapper = TokenSHAPWrapper(
            model=model,
            tokenizer=tokenizer,
            max_length=self.max_length
        )
        return TokenSHAP(wrapper, StringSplitter())

    def explain(self, text: str, predict_fn: callable) -> Dict:
        try:
            # Get TokenSHAP analysis
            token_shap_result = predict_fn.analyze(
                text, 
                sampling_ratio=self.sampling_ratio,
                print_highlight_text=True  
            )
            
            # Generate visualization
            visualization = predict_fn.print_colored_text()
            
            return {
                'token_shap_values': token_shap_result,
                'visualization': visualization  # Store the visualization
            }
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class TokenSHAPWrapper:
    """Wrapper class for models to work with TokenSHAP with memory management"""
    
    def __init__(self, model, tokenizer, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # pad token 
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
    def __call__(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, input_ids, attention_mask=None) -> str:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if isinstance(input_ids, str):
                inputs = self.tokenizer(
                    input_ids,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

            # Move to device
            input_ids = input_ids.to(self.model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            else:
                attention_mask = torch.ones_like(input_ids).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length
                )

            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()