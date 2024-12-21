from typing import Dict, List, Optional
import torch
from datasets import load_dataset
from . import ProcessingStrategy
from ..utils.explanation_saver import save_explanation
from ..explainers.lime_explainer import LIMEExplainer
from ..explainers.shap_explainer import SHAPExplainer

class EncoderStrategy(ProcessingStrategy):
    """Strategy for handling BERT-style encoder models"""
    
    def __init__(self, model, tokenizer, explainer_types: Optional[List[str]] = None):
        super().__init__(model, tokenizer, explainer_types)
        self.device = next(model.parameters()).device  # Get model's device
        self.background_dataset = None
        if explainer_types and 'shap' in explainer_types:
            # Load background dataset for SHAP
            dataset = load_dataset("HiTZ/casimedicos-exp", "en")
            train_data = dataset['train']
            self.background_dataset = [case['full_question'] for case in list(train_data)[:100]]  # First 100 samples
            
    def _setup_explainers(self) -> Dict:
        """Setup requested explainers for encoder models"""
        explainers = {}
        
        if not self.explainer_types:  # If no explainers specified, return empty dict
            return explainers
            
        # Setup only requested explainers
        if 'lime' in self.explainer_types:
            explainers['lime'] = LIMEExplainer()
            
        if 'shap' in self.explainer_types:
            explainers['shap'] = SHAPExplainer()
            
        return explainers
        
    def preprocess(self, case: Dict) -> Dict:
        """
        Preprocess a single case for BERT-style models.
        
        Args:
            case: Single case from dataset with 'full_question' and 'options'
            
        Returns:
            Dict containing processed input ready for model
        """
        try:
            # Extract valid options (filtering out None values)
            valid_options = {k: v for k, v in case['options'].items() if v is not None}
            option_texts = list(valid_options.values())
            option_keys = list(valid_options.keys())
            
            # Tokenize the question and options
            encodings = self.tokenizer(
                [case['full_question']] * len(option_texts),  # Repeat question for each option
                option_texts,  
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Prepare model inputs
            processed = {
                'input_ids': encodings['input_ids'].unsqueeze(0),
                'attention_mask': encodings['attention_mask'].unsqueeze(0),
                'option_keys': option_keys,
                'original': case  # Keep original case for reference
            }
            
            return processed
            
        except Exception as e:
            raise RuntimeError(f"Error preprocessing case {case.get('id', 'unknown')}: {str(e)}")
        
    def predict(self, processed_case: Dict) -> Dict:
        """
        Generate prediction for processed case
        
        Args:
            processed_case: Output from preprocess method
            
        Returns:
            Dict containing predictions and probabilities
        """
        try:
            input_ids = processed_case['input_ids']
            attention_mask = processed_case['attention_mask']
            option_keys = processed_case['option_keys']
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                
            predicted_idx = probs.argmax().item()
            predicted_option = option_keys[predicted_idx]
            correct_option = processed_case['original']['correct_option']
            is_correct = str(predicted_option) == str(correct_option)
            
            result = {
                'prediction': predicted_option,
                'confidence': float(probs[predicted_idx].cpu()),
                'probabilities': probs.cpu().numpy(),
                'option_keys': option_keys,
                'original': processed_case['original'],
                'is_correct': is_correct
            }

            del outputs
            torch.cuda.empty_cache()
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")       

    def explain(self, processed_case: Dict, prediction: Dict) -> Dict:
        """Generate explanations for the prediction using configured explainers"""
        explanations = {}
        
        if not self.explainers:  # Skip if no explainers configured
            return explanations
            
        try:
            # Basic case info for saving explanations
            case_info = {
                'id': prediction['original'].get('id', 'unknown'),
                'question': prediction['original']['full_question'],
                'correct_option': prediction['original']['correct_option'],
                'predicted_option': prediction['prediction'],
                'options': prediction['option_keys'],
                'probabilities': prediction['probabilities'],
                'is_correct': str(prediction['prediction']) == str(prediction['original']['correct_option'])
            }
            
            # For each configured explainer
            for name, explainer in self.explainers.items():
                try:
                    # Create predict function for explainer
                    predict_fn = explainer.create_explainer_function(
                        options=prediction['option_keys'],
                        tokenizer=self.tokenizer,
                        model=self.model,
                        device=self.device
                    )
                    
                    # Get explanation
                    exp = explainer.explain(
                        question=prediction['original']['full_question'],
                        predict_fn=predict_fn,
                        class_names=[f"Option {k}" for k in prediction['option_keys']],
                        target_label=prediction['predicted_idx'],
                        background_dataset=self.background_dataset if name == 'shap' else None
                    )
                    
                    # Save explanation
                    save_dir = save_explanation(exp, case_info, explainer_type=name)
                    explanations[name] = {
                        'exp': exp,
                        'save_dir': save_dir
                    }
                    
                except Exception as e:
                    print(f"Error in {name} explainer: {str(e)}")
                    
            return explanations
                
        except Exception as e:
            raise RuntimeError(f"Explanation generation failed: {str(e)}") 