from typing import Dict, Generator, Optional, List
from datasets import load_dataset
from .strategies import ModelType, StrategyFactory
import torch
from .context2nl import ContextGroundedExplainer
class Pipeline:
    def __init__(self, 
                 model_name: str, 
                 model_type: ModelType,
                 explainer_types: Optional[List[str]] = None,
                 device: str = None,
                 sampling_ratio: float = 0.0,
                 max_length: int = 256,
                 token_shap_max_tokens: int = 32,
                 explanation_max_tokens: int = 256
                 ):
        """
        Initialize pipeline with specified model and explainers
        Args:
            model_name (str): HuggingFace model name or path.
            model_type (ModelType): Type of model (ENCODER/DECODER).
            explainer_types (Optional[List[str]]): List of explainer types to use ('lime', 'shap', 'TokenShap', etc.).
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to None, which auto-selects based on availability.
            sampling_ratio (float, optional): Sampling ratio for TokenSHAP (applicable for decoder only). Defaults to 0.0.
            max_length (int, optional): Maximum sequence length (applicable for decoder only). Defaults to 256.
            token_shap_max_tokens (int, optional): Maximum number of tokens for TokenSHAP. Defaults to 32.
            explanation_max_tokens (int, optional): Maximum number of tokens for explanations. Defaults to 256.

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
            explanation_max_tokens=explanation_max_tokens
        )
        # For tracking metrics
        self.metrics = {
            'total_processed': 0,
            'correct_predictions': 0,
            'errors': []
        }
        self.explanation_max_tokens=explanation_max_tokens
        # print("[DEBUG] Metrics initialized.")
        self.context_explainer = ContextGroundedExplainer()
        print("[DEBUG] Pipeline Initialized.")
        
    def process_dataset(self, split: str = 'validation', limit: int = None, generate_natural_language_explanation = False, explainer_model_type:str = "mixtral-8x7b-32768", explanation_max_tokens=None, llm_prompt_template = None) -> Generator:
        """
        Process the dataset with the specified explainers.

        Args:
            split (str): The dataset split to process ('train', 'validation', or 'test'). Defaults to 'validation'.
            limit (int, optional): The number of cases to process. If None, process all cases. Defaults to None.
            generate_natural_language_explanation (bool, optional): Whether to generate natural language explanations. Defaults to False.
            explainer_model_type (str, optional): The type of explainer model to use available from groq. Defaults to "mixtral-8x7b-32768".
            explanation_max_tokens (int, optional): The maximum number of tokens for explanations for the context grounded explanation step.

        Yields:
            Generator: A generator that yields processed cases.
        """
        dataset = load_dataset("HiTZ/casimedicos-exp", "en")[split]
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            print(f"Processing {limit} cases...")
        else:
            print(f"Processing {len(dataset)} cases...")

        for case in dataset:
            #try:
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

                if generate_natural_language_explanation:
                    if explanation_max_tokens == None:
                        explanation_max_tokens = self.explanation_max_tokens
                    context_grounded_reasoning = self.context_explainer.generate_response(case, explanations, prediction, explainer_model_type, explanation_max_tokens, llm_prompt_template)
                    yield {
                        'prediction': prediction,
                        'explanations': explanations,
                        'original': case,
                        'context_grounded_reasoning': context_grounded_reasoning
                    }
                else:
                    yield {
                        'prediction': prediction,
                        'explanations': explanations,
                        'original': case
                    }
                # Maybe periodic cleanup
                if self.metrics['total_processed'] % 10 == 0:
                    torch.cuda.empty_cache()

            #except Exception as e:
            #    self.metrics['errors'].append(str(e))
            #    yield {'error': str(e), 'original': case}

        # Print final metrics
        print(f"\nProcessing complete!")
        print(f"Total processed: {self.metrics['total_processed']}")
        print(f"Correct predictions: {self.metrics['correct_predictions']}")
        if self.metrics['total_processed'] > 0:
            accuracy = self.metrics['correct_predictions'] / self.metrics['total_processed']
            print(f"Accuracy: {accuracy:.2%}")