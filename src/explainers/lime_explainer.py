from typing import List, Dict, Any, Callable
import numpy as np
import torch
import gc
from lime.lime_text import LimeTextExplainer
from . import BaseExplainer  

class LIMEExplainer(BaseExplainer):
    """LIME-based explanation generator for clinical text classification."""
    
    def __init__(self, 
                 batch_size: int = 100,
                 num_samples: int = 1000,
                 num_features: int = 20):
        """
        Initialize LIME explainer with configuration.
        
        Args:
            batch_size: Number of samples to process in each batch
            num_samples: Number of neighborhood samples for LIME
            num_features: Maximum number of features to include in explanation
        """
        super().__init__()  
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_features = num_features
        
    def create_explainer_function(self, options: List[str], tokenizer: Any, model: Any, device: str):
        """
        Creates a batched prediction function closure for LIME.
        
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

        def prediction_function(perturbed_questions):
            if not isinstance(perturbed_questions, (list, np.ndarray)):
                raise TypeError(f"Expected list or numpy array, got {type(perturbed_questions)}")

            num_perturbations = len(perturbed_questions)
            num_options = len(options)
            all_probs = []
            
            # Calculate number of batches
            num_batches = (num_perturbations + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                # Calculate batch start and end indices
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, num_perturbations)
                batch_questions = perturbed_questions[start_idx:end_idx]
                current_batch_size = len(batch_questions)
                
                # Create pairs for current batch
                batch_paired_questions = np.repeat(batch_questions, num_options)
                batch_paired_options = np.tile(options, current_batch_size)
                
                try:
                    # Tokenize batch
                    encodings = tokenizer(
                        batch_paired_questions.tolist(),
                        batch_paired_options.tolist(),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Reshape for the model
                    input_ids = encodings['input_ids'].view(current_batch_size, num_options, -1).to(device)
                    attention_mask = encodings['attention_mask'].view(current_batch_size, num_options, -1).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        batch_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                    
                    # Append batch probabilities
                    all_probs.append(batch_probs)
                    
                finally:
                    # Clear GPU cache after each batch
                    del input_ids, attention_mask, outputs
                    torch.cuda.empty_cache()
            
            # Concatenate all batch probabilities
            return np.vstack(all_probs)
        
        return prediction_function
    
    def explain(self, question: str, predict_fn: callable, class_names: List[str], target_label: int, **kwargs) -> Dict:
        target_label = int(target_label)
        explainer = None
        exp = None
        try:
            print("Creating LIME explainer...")
            explainer = LimeTextExplainer(
                class_names=class_names,
                verbose=True,
                split_expression='\s+',
                bow=False,
                mask_string='[MASK]',
                random_state=9
            )
            
            print(f"Generating explanation for target_label: {target_label}")
            exp = explainer.explain_instance(
                question,
                predict_fn,
                num_features=self.num_features,
                labels=[target_label],
                num_samples=self.num_samples
            )
            if exp is not None:
                exp.show_in_notebook(text=True)

            return exp
                
        except Exception as e:
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            print(f"Error occurred in explain method with:")
            print(f"- question length: {len(question)}")
            print(f"- class_names: {class_names}")
            print(f"- target_label: {target_label}")
            raise RuntimeError(f"LIME explanation failed: {type(e).__name__}: {str(e)}") from e
        
        finally:
            # Cleanup
            del explainer
            del exp
            gc.collect()
            torch.cuda.empty_cache()

    def _monitor_resources(self, stage: str = ""):
        """Monitor CPU and GPU memory usage."""
        import psutil
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
