from typing import Dict, Generator, Optional, List
from datasets import load_dataset
from .strategies import ModelType, StrategyFactory
import torch

class Pipeline:
    def __init__(self, 
                 model_name: str, 
                 model_type: ModelType,
                 explainer_types: Optional[List[str]] = None,
                 device: str = None,
                 sampling_ratio: float = 0.0,
                 max_length: int = 256,
                 token_shap_max_tokens: int = 32,
                 explanation_max_tokens: int = 256,):
        """
        Initialize pipeline with specified model and explainers
        Args:
            model_name: HuggingFace model name/path
            model_type: Type of model (ENCODER/DECODER)
            explainer_types: List of explainer types to use ('lime', 'shap', 'TokenShap', etc.)
            sampling_ratio: Sampling ratio for TokenSHAP (decoder only)
            max_length: Maximum sequence length (decoder only)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n=== Pipeline Initialization ===")
        print(f"Model Name: {model_name}")
        print(f"Model Type: {model_type}")
        print(f"Device: {device}")
        print("=============================\n")
    
        self.strategy = StrategyFactory.create_strategy(
            model_type=model_type,
            model_name=model_name,
            explainer_types=explainer_types,
            device=device,
            max_length=max_length,
            sampling_ratio=sampling_ratio,
            token_shap_max_tokens=token_shap_max_tokens,
            explanation_max_tokens=explanation_max_tokens,
        )
        # For tracking metrics
        self.metrics = {
            'total_processed': 0,
            'correct_predictions': 0,
            'errors': []
        }
        
    def process_dataset(self, split: str = 'validation', limit: int = None) -> Generator:
        """Process dataset with specified explainers
        
        Args:
            split: 'train', 'validation', or 'test'
            limit: number of cases to process
        """
        dataset = load_dataset("HiTZ/casimedicos-exp", "en")[split]
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            print(f"Processing {limit} cases...")
        else:
            print(f"Processing {len(dataset)} cases...")

        for case in dataset:
            try:
                print("\n" + "="*80)
                print(f"📋 Case ID: {case.get('id', 'unknown')}")
                print(f"Type: {case['type']}")

                processed = self.strategy.preprocess(case)
                prediction = self.strategy.predict(processed)
                explanations = self.strategy.explain(processed, prediction)
                
                # Update metrics
                self.metrics['total_processed'] += 1
                if prediction['is_correct']:
                    self.metrics['correct_predictions'] += 1

                yield {
                    'prediction': prediction,
                    'explanations': explanations,
                    'original': case
                }
                # Maybe periodic cleanup
                if self.metrics['total_processed'] % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                self.metrics['errors'].append(str(e))
                yield {'error': str(e), 'original': case}

        # Print final metrics
        print(f"\nProcessing complete!")
        print(f"Total processed: {self.metrics['total_processed']}")
        print(f"Correct predictions: {self.metrics['correct_predictions']}")
        if self.metrics['total_processed'] > 0:
            accuracy = self.metrics['correct_predictions'] / self.metrics['total_processed']
            print(f"Accuracy: {accuracy:.2%}")