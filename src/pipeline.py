from typing import Dict, List, Generator, Union, Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from .explainers import BaseExplainer, ExplainerType
from .explainers.lime_explainer import LIMEExplainer
from .explainers.shap_explainer import SHAPExplainer
from .utils.explanation_saver import save_explanation


class Pipeline:
    def __init__(self, 
                 model_name: str,
                 explainer_types: Optional[Union[ExplainerType, List[ExplainerType]]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 background_samples: int = 100):
        """
        Initialize pipeline with specified explainer types.
        
        Args:
            model_name: HuggingFace model name
            explainer_types: ExplainerType.LIME, ExplainerType.SHAP, or list of them
            device: 'cuda' or 'cpu'
            background_samples: Number of background samples for SHAP

        """
        self.model_name = model_name
        self.device = device
        self.background_samples = background_samples
        self.background_dataset = None  # Will be populated in setup()
        
        if isinstance(explainer_types, ExplainerType):
            explainer_types = [explainer_types]
            
        self.explainer_types = explainer_types
        self.explainers = self._setup_explainers() if explainer_types else {}
        
        self.tokenizer = None
        self.model = None
        
        self.metrics = {
            'total_processed': 0,
            'correct_predictions': 0,
            'errors': []
        }
    
    def _setup_explainers(self) -> Dict:
        """Initialize requested explainers"""
        explainers = {}
        if not self.explainer_types:
            return explainers
            
        for explainer_type in self.explainer_types:
            if explainer_type == ExplainerType.LIME:
                explainers['lime'] = LIMEExplainer()
            elif explainer_type == ExplainerType.SHAP:
                explainers['shap'] = SHAPExplainer()

        return explainers

    def setup(self):
        """Initialize model and tokenizer"""
        try:
            print(f"Loading model and tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMultipleChoice.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            # Load background dataset for SHAP if SHAP explainer is requested
            if any(exp_type == ExplainerType.SHAP for exp_type in self.explainer_types):
                print("Loading background dataset for SHAP...")
                dataset = load_dataset("HiTZ/casimedicos-exp", "en")
                train_data = dataset['train']
                self.background_dataset = [case['full_question'] 
                                         for case in list(train_data)[:self.background_samples]]
                print(f"Loaded {len(self.background_dataset)} background samples")
            
            return self
        except Exception as e:
            raise RuntimeError(f"Pipeline setup failed: {str(e)}")

    def _preprocess_generator(self, cases: List[Dict]) -> Generator:
        """Generate preprocessed cases one at a time"""
        for case in cases:
            try:
                valid_options = {k: v for k, v in case['options'].items() if v is not None}
                option_texts = list(valid_options.values())
                option_keys = list(valid_options.keys())
                
                encodings = self.tokenizer(
                    [case['full_question']] * len(option_texts),
                    option_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                processed = {
                    'input_ids': encodings['input_ids'].unsqueeze(0),
                    'attention_mask': encodings['attention_mask'].unsqueeze(0),
                    'option_keys': option_keys,
                    'original': case
                }
                
                yield processed
                
            except Exception as e:
                print(f"Error preprocessing case {case.get('id', 'unknown')}: {str(e)}")
                yield {'error': str(e), 'original': case}


    def _generate_explanations(self, case: Dict, predicted_idx: int) -> Dict:
        """Generate explanations using all active explainers"""
        explanations = {}
        
        for explainer_name, explainer in self.explainers.items():
            try:
                predict_fn = explainer.create_explainer_function(
                    options=case['option_keys'],
                    tokenizer=self.tokenizer,
                    model=self.model,
                    device=self.device
                )
                
                exp = explainer.explain(
                    question=case['original']['full_question'],
                    predict_fn=predict_fn,
                    class_names=[f"Option {k}" for k in case['option_keys']],
                    target_label=int(predicted_idx),
                    background_dataset=self.background_dataset if explainer_name == 'shap' else None
                )
                case_info = {
                    'id': case['original'].get('id', 'unknown'),
                    'question': case['original']['full_question'],
                    'correct_option': case['original']['correct_option'],
                    'predicted_option': case['option_keys'][predicted_idx],
                    'options': case['option_keys'],
                    'probabilities': case['probabilities'].tolist() if 'probabilities' in case else None,
                    'is_correct': str(case['option_keys'][predicted_idx]) == str(case['original']['correct_option'])
                }
                
                # Save explanation
                save_dir = save_explanation(exp, case_info, explainer_type=explainer_name)
                print(f"\nExplanations saved.....")
            
                explanations[explainer_name] = {
                    'exp': exp,
                    'save_dir': save_dir
                }
                
            except Exception as e:
                raise RuntimeError(f"generate_explanation failed: {str(e)}")
        
        return explanations

    def _predict_generator(self, processed_cases: Generator) -> Generator:
        """Generate predictions with explanations if requested"""
        for case in processed_cases:
            if 'error' in case:
                yield case
                continue
                
            try:
                print("\n" + "="*80)
                print(f"📋 Case ID: {case['original'].get('id', 'unknown')}")
                print(f"Type: {case['original']['type']}")
                
                print("\n📝 Question:")
                print(case['original']['full_question'])
                
                print("\n🔤 Options:")
                valid_options = {k: v for k, v in case['original']['options'].items() if v is not None}
                for key, text in valid_options.items():
                    print(f"Option {key}: {text}")

                input_ids = case['input_ids'].to(self.device)
                attention_mask = case['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                
                probabilities = probs.cpu().numpy()
                predicted_idx = probabilities.argmax()
                predicted_option = case['option_keys'][predicted_idx]

                # Verify correct option is in our valid options
                correct_option = case['original']['correct_option']
                
                self.metrics['total_processed'] += 1
                is_correct = str(predicted_option) == str(correct_option)

                if is_correct:
                    self.metrics['correct_predictions'] += 1
                
                # Print prediction results
                print("\n🎯 Prediction Results:")
                print(f"Model predicted: Option {predicted_option}")
                print(f"Correct answer: Option {correct_option}")
                print(f"Status: {'✅ Correct' if is_correct else '❌ Wrong'}")
                print(f"Confidence: {float(probabilities[predicted_idx]):.2%}")
                
                print("\nProbabilities for each valid option:")
                for key, prob in zip(case['option_keys'], probabilities):
                    print(f"Option {key}: {prob:.2%}")
                
                print("="*80)
                
                result = {
                    'prediction': predicted_option,
                    'confidence': float(probabilities[predicted_idx]),
                    'probabilities': probabilities,
                    'option_keys': case['option_keys'],
                    'original': case['original'],
                    'is_correct': str(predicted_option) == str(case['original']['correct_option'])
                }
                
                # Generate explanations if explainers exist
                if self.explainers:
                    explanations = self._generate_explanations(case, predicted_idx)
                    result['explanations'] = explanations
                
                yield result
                    
            except Exception as e:
                print(f"\n❌ Error predicting case {case['original'].get('id', 'unknown')}: {str(e)}")
                yield {'error': str(e), 'original': case['original']}

    def process_dataset(self, 
                       split: str = 'validation', 
                       limit: int = None,
                       explainer_types: Optional[Union[ExplainerType, List[ExplainerType]]] = None) -> List[Dict]:
        """
        Process dataset with specified explainers.
        
        Args:
            split: 'train', 'validation', or 'test'
            limit: number of cases to process
            explainer_types: Override default explainers for this run
        """
        # Temporarily override explainers if requested
        original_explainers = self.explainers
        if explainer_types:
            if isinstance(explainer_types, ExplainerType):
                explainer_types = [explainer_types]
            self.explainer_types = explainer_types
            self.explainers = self._setup_explainers()
        
        try:
            print(f"Loading CasiMedicos-Arg {split} split...")
            dataset = load_dataset("HiTZ/casimedicos-exp", "en")[split]
            
            # If limit is set, only take that many cases
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
                print(f"Processing {limit} cases...")
            else:
                print(f"Processing {len(dataset)} cases...")
            
            # Process through pipeline
            processed_cases = self._preprocess_generator(dataset)
            results = list(self._predict_generator(processed_cases))
            
            # Print metrics
            print(f"\nProcessing complete!")
            print(f"Total processed: {self.metrics['total_processed']}")
            print(f"Correct predictions: {self.metrics['correct_predictions']}")
            if self.metrics['total_processed'] > 0:
                accuracy = self.metrics['correct_predictions'] / self.metrics['total_processed']
                print(f"Accuracy: {accuracy:.2%}")

            return results
        
        finally:
            # Restore original explainers
            if explainer_types:
                self.explainers = original_explainers
