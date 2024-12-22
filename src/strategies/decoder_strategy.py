from typing import Dict, List, Optional
import torch
import gc
from . import ProcessingStrategy
from ..utils.explanation_saver import save_explanation
from ..explainers.token_shap_explainer import TokenSHAPExplainer

class DecoderStrategy(ProcessingStrategy):
    """Strategy for handling GPT/Mistral style decoder models"""
    
    def __init__(self, 
                 model, 
                 tokenizer, 
                 explainer_types: Optional[List[str]] = None,
                 max_length: int = 256,
                 sampling_ratio: float = 0.0):
        self.device = next(model.parameters()).device
        self.max_length = max_length
        self.sampling_ratio = sampling_ratio
        super().__init__(model, tokenizer, explainer_types)
    def _setup_explainers(self) -> Dict:
        """Setup TokenSHAP explainer"""
        explainers = {}
        
        if 'token_shap' in self.explainer_types:
            explainers['token_shap'] = TokenSHAPExplainer(
                sampling_ratio=self.sampling_ratio,
                max_length=self.max_length
            )
        
        return explainers

    def preprocess(self, case: Dict) -> Dict:
        """Preprocess a clinical case for decoder models"""
        try:
            valid_options = {k: v for k, v in case['options'].items() if v is not None}
            
            prompt = (
                f"Case:\n{case['full_question']}\n\n"
                f"Options:\n" +
                "\n".join(f"{k}. {v}" for k, v in valid_options.items()) +
                "\n\nProvide the option number and brief explanation."
            )
                
            return {
                'prompt': prompt,
                'options': valid_options,
                'original': case
            }
            
        except Exception as e:
            raise RuntimeError(f"Error preprocessing case {case.get('id', 'unknown')}: {str(e)}")

    def predict(self, processed_case: Dict) -> Dict:
        """Generate model prediction"""
        try:
            # Create predict function for current case
            predict_fn = self.explainers['token_shap'].create_explainer_function(
                self.model, 
                self.tokenizer
            )
            
            # Get model response
            response = predict_fn(processed_case['prompt'])
            
            result = {
                'response': response,
                'original': processed_case['original'],
                'options': processed_case['options'],
                'predict_fn': predict_fn  # Save for explanation phase
            }
            
            return result
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def explain(self, processed_case: Dict, prediction: Dict) -> Dict:
        """Generate explanations using TokenSHAP"""
        explanations = {}
        
        try:
            if 'token_shap' in self.explainers:
                # Generate TokenSHAP explanation
                explanation_result = self.explainers['token_shap'].explain(
                    processed_case['prompt'],
                    prediction['predict_fn']
                )
                
                # Prepare combined results
                explanation_data = {
                    'token_shap_values': explanation_result['token_shap_values'],
                    'visualization': explanation_result['visualization'],
                    'model_response': prediction['response'],
                    'case_id': processed_case['original']['id'],
                    'options': processed_case['options']
                }
                
                # Save for future use
                save_path = save_explanation(
                    exp=explanation_data,
                    case_info=processed_case['original'],
                    explainer_type='decoder_combined'
                )
                explanations['token_shap'] = {
                    'explanation': explanation_result,
                    'save_path': save_path
                }

            return explanations
    
        finally:
            # Cleanup
            del prediction['predict_fn']  # Remove the prediction function
            torch.cuda.empty_cache()
            gc.collect()