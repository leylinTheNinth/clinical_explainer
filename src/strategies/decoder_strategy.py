from typing import Dict
import torch
from token_shap import StringSplitter, TokenSHAP
from . import ProcessingStrategy
from ..explainers import TokenSHAPModel
from ..utils import save_decoder_outputs
from ..templates import PromptTemplateFactory  # Add this import


class DecoderStrategy(ProcessingStrategy):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 max_length: int = 256,
                 sampling_ratio: float = 0.0,
                 token_shap_max_tokens: int = 32,
                 explanation_max_tokens: int = 256):
        super().__init__(model, tokenizer, None)  # No explainer_types needed
        self.max_length = max_length
        self.explanation_max_tokens = explanation_max_tokens
        self.token_shap_max_tokens = token_shap_max_tokens
        self.sampling_ratio = sampling_ratio
        self.token_shap_wrapper = TokenSHAPModel(
            model, 
            tokenizer, 
            max_length,
            max_new_tokens=token_shap_max_tokens)
        
        print(f"\n=== Strategy Configuration ===")
        print(f"Explanation Max Tokens: {explanation_max_tokens}")
        print(f"TokenSHAP Max Tokens: {token_shap_max_tokens}")
        print(f"Sampling Ratio: {sampling_ratio}")
        print("===========================\n")

        # Get model name and setup template
        model_name = model.config.name_or_path
        self.template = PromptTemplateFactory.create_template(model_name)
        print(f"\n=== Template Configuration ===")
        print(f"Model Name: {model_name}")
        print(f"Selected Template: {self.template.__class__.__name__}")
        print(f"User Prefix: {repr(self.template.user_prefix)}")
        print(f"User Suffix: {repr(self.template.user_suffix)}")
        print(f"Assistant Prefix: {repr(self.template.assistant_prefix)}")
        print("============================\n")
        
    def preprocess(self, case: Dict) -> Dict:
        """Prepare prompts using model-specific template"""
        # Filter valid options
        options = {k: v for k, v in case['options'].items() if v is not None}

        # Use template to format prompt
        prediction_prompt = self.template.format_prompt(case)
        
        print("\n=== Prompt Preview ===")
        print("First 200 characters of formatted prompt:")
        print(prediction_prompt[:200] + "...")
        print("=====================\n")
        
        return {
            'prediction_prompt': prediction_prompt,
            'token_shap_prompt': prediction_prompt,
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
                    max_new_tokens=self.explanation_max_tokens,
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
            print("Running TokenSHAP analysis...")
            print(f"Using sampling ratio: {self.sampling_ratio}")
            print(f"Max tokens: {self.token_shap_max_tokens}")
        
            explanation = token_shap.analyze(
                processed_case['token_shap_prompt'],
                sampling_ratio=self.sampling_ratio,  
                print_highlight_text=True
            )

            model_name = self.model.config.name_or_path

            print(f"\nTokenSHAP Analysis:")
            token_shap.print_colored_text()
            print("TokenSHAP Analysis Complete")
            print("=========================\n")

            save_path = save_decoder_outputs(
                token_shap_exp=token_shap,
                case_info=prediction['original'],  # Using the original case info
                prediction=prediction,
                model_name=model_name
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