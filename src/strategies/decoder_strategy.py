from typing import Dict
import torch
from token_shap import StringSplitter, TokenSHAP
from . import ProcessingStrategy
from ..explainers import TokenSHAPModel
from ..utils import save_decoder_outputs

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
        
        # Base prompt structure that's common to both
        base_prompt = (
            "You are a medical expert analyzing this clinical case.\n\n"
            "Clinical Case:\n"
            f"{case['full_question']}\n\n"
            "Options:\n"
        )
        for opt_num, opt_text in options.items():
            base_prompt += f"{opt_num}: {opt_text}\n"

        # For prediction 
        prediction_prompt = base_prompt + (
            "\nAnalyze this case and provide:\n"
            "1. Your selected option (number only)\n"
            "2. Medical reasoning behind your decision\n"
            "Your Response: "
        )

        # For TokenSHAP, keep it focused on decision
        token_shap_prompt = base_prompt + "\nSelect the most appropriate option (number only).\nYour Response: "


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
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nModel Response:\n{response}\n")

            return {
                'response': response,
                'original': processed_case['original'],
                'is_correct': False  # Temporary fix
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
                sampling_ratio=0.0, # if increased processing time will also increase 
                print_highlight_text=True
            )

            print(f"\nTokenSHAP Analysis:")
            token_shap.print_colored_text()

            save_path = save_decoder_outputs(
                token_shap_exp=token_shap,
                case_info=prediction['original'],  # Using the original case info
                prediction=prediction
            )
        
            return {
                'token_shap': {
                    'exp': token_shap,
                    'shapley_values': token_shap.shapley_values,
                    'save_path': save_path
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"TokenSHAP analysis failed: {str(e)}")