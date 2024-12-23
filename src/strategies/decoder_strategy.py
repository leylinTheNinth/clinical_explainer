from typing import Dict
import torch
from token_shap import StringSplitter, TokenSHAP
from . import ProcessingStrategy
from ..explainers import TokenSHAPModel

class DecoderStrategy(ProcessingStrategy):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 max_length: int = 256):
        super().__init__(model, tokenizer, None)  # No explainer_types needed
        self.max_length = max_length
        self.token_shap_wrapper = TokenSHAPModel(model, tokenizer, max_length)
        
    def preprocess(self, case: Dict) -> Dict:
        """Prepare prompts for both prediction and TokenSHAP"""
        # Filter valid options
        options = {k: v for k, v in case['options'].items() if v is not None}
        
        # Prompt for prediction (with explanation request)
        prediction_prompt = (
            "Clinical Case:\n"
            f"{case['full_question']}\n\n"
            "Options:\n"
        )
        for opt_num, opt_text in options.items():
            prediction_prompt += f"{opt_num}: {opt_text}\n"
        prediction_prompt += "\nWhich option is most appropriate? Explain your choice."

        # Prompt for TokenSHAP (just the case and options)
        token_shap_prompt = (
            "Clinical Case:\n"
            f"{case['full_question']}\n\n"
            "Options:\n"
        )
        for opt_num, opt_text in options.items():
            token_shap_prompt += f"{opt_num}: {opt_text}\n"
        token_shap_prompt += "\nWhich option is most appropriate?"

        return {
            'prediction_prompt': prediction_prompt,
            'token_shap_prompt': token_shap_prompt,
            'options': options,
            'original': case
        }

    def predict(self, processed_case: Dict) -> Dict:
        """Get model's prediction with explanation"""
        try:
            inputs = self.tokenizer(
                processed_case['prediction_prompt'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'response': response,
                'original': processed_case['original']
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def explain(self, processed_case: Dict, prediction: Dict) -> Dict:
        """Use TokenSHAP to analyze important tokens in option selection"""
        try:
            splitter = StringSplitter()
            token_shap = TokenSHAP(self.token_shap_wrapper, splitter)
            
            explanation = token_shap.analyze(
                processed_case['token_shap_prompt'],
                sampling_ratio=0.0,
                print_highlight_text=True
            )
            
            return {'token_shap': explanation}
            
        except Exception as e:
            raise RuntimeError(f"TokenSHAP analysis failed: {str(e)}")