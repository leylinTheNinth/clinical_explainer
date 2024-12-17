from typing import List, Dict, Any, Callable
import numpy as np
import torch
import gc
import shap
import scipy as sp
from . import BaseExplainer
import psutil


class SHAPExplainer(BaseExplainer):
    """SHAP-based explanation generator for clinical text classification."""
    def __init__(self, 
                 batch_size: int = 64,
                 background_samples: int = 100):
        super().__init__()
        self.batch_size = batch_size
        self.background_samples = background_samples
        self.explainer = None
        self.tokenizer = None  # Will be set when creating explainer function
        
    def create_explainer_function(self, options: List[str], tokenizer: Any, model: Any, device: str):
        """
        Creates a batched prediction function closure for SHAP.
        
        Args:
            options: List of option texts
            tokenizer: HuggingFace tokenizer
            model: HuggingFace model
            device: torch device
            
        Returns:
            function: Prediction function that processes samples in batches
        """
        if not options:
            raise ValueError("Options list cannot be empty")
        if not tokenizer:
            raise ValueError("Tokenizer cannot be None")
        if not model:
            raise ValueError("Model cannot be None")

        self.tokenizer = tokenizer  # Store tokenizer for explainer initialization

        def prediction_function(perturbed_questions):
            if not isinstance(perturbed_questions, (list, np.ndarray)):
                raise TypeError(f"Expected list or numpy array, got {type(perturbed_questions)}")

            outputs = []
            
            # Process in batches
            for i in range(0, len(perturbed_questions), self.batch_size):
                batch = perturbed_questions[i:i + self.batch_size]
                
                # For each perturbed text, pair with all options
                batch_questions = np.repeat(batch, len(options))
                batch_options = np.tile(options, len(batch))
                
                try:
                    # Tokenize batch
                    encodings = tokenizer(
                        batch_questions.tolist(),
                        batch_options.tolist(),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Reshape for model
                    input_ids = encodings['input_ids'].view(-1, len(options), encodings['input_ids'].size(-1)).to(device)
                    attention_mask = encodings['attention_mask'].view(-1, len(options), encodings['attention_mask'].size(-1)).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs_batch = model(input_ids=input_ids, attention_mask=attention_mask)
                        probs_batch = torch.softmax(outputs_batch.logits, dim=1)
                        # Get probability for target option only
                        target_probs = probs_batch[:, 0].cpu().numpy()  # For target option
                        outputs.extend(target_probs)
                        
                finally:
                    # Clear GPU cache after each batch
                    del input_ids, attention_mask, outputs_batch
                    torch.cuda.empty_cache()
            
            outputs = np.array(outputs)
            # Convert to log odds for SHAP
            return sp.special.logit(outputs)
        
        return prediction_function
        
    
    def initialize_explainer(self, predict_fn: callable, background_dataset: List[str]):
        """Initialize SHAP explainer with background dataset"""
        if self.explainer is None:
            print("Creating SHAP explainer...")
            #self._monitor_resources("Before SHAP Explainer Creation")
            
            if not self.tokenizer:
                raise ValueError("Tokenizer not set. Call create_explainer_function first")
                
            self.explainer = shap.Explainer(predict_fn, 
                                          masker=shap.maskers.Text(self.tokenizer),  
                                          background=background_dataset)
    
    def explain(self, 
                question: str, 
                predict_fn: callable, 
                class_names: List[str], 
                target_label: int,
                **kwargs) -> Dict:
        """
        Generate SHAP explanation for a given question.
        
        Args:
            question: The question text to explain
            predict_fn: The prediction function
            class_names: List of class names
            target_label: The target class to explain
            **kwargs: Additional arguments including:
                     - background_dataset: Required list of background texts for SHAP
            
        Returns:
            Dict: SHAP explanation object
        """
        shap_values = None

        background_dataset = kwargs.get('background_dataset')
        if background_dataset is None:
            raise ValueError("background_dataset is required for SHAP")

        try:
            # Initialize explainer if not already done
            if self.explainer is None:
                self.initialize_explainer(predict_fn, background_dataset)
            
            print(f"Generating explanation for target_label: {target_label}")
            
            # Calculate SHAP values using existing explainer
            shap_values = self.explainer([question])
            
            return shap_values
            
        except Exception as e:
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            print(f"Error occurred in explain method with:")
            print(f"- question length: {len(question)}")
            print(f"- class_names: {class_names}")
            print(f"- target_label: {target_label}")
            print(f"- background dataset size: {len(background_dataset)}")
            raise RuntimeError(f"SHAP explanation failed: {type(e).__name__}: {str(e)}") from e
            
        finally:
            del shap_values
            gc.collect()
            torch.cuda.empty_cache()

    def _monitor_resources(self, stage: str = ""):
        """Monitor CPU and GPU memory usage."""
        print(f"\n--- Resource Usage at {stage} ---")
        # CPU memory
        cpu_mem = psutil.virtual_memory().percent
        print(f"CPU Memory Usage: {cpu_mem}%")
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated() / 1e6
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e6
            print(f"GPU Memory Allocated: {gpu_mem_alloc:.2f} MB")
            print(f"GPU Memory Reserved: {gpu_mem_reserved:.2f} MB")
        print("-----------------------------")
