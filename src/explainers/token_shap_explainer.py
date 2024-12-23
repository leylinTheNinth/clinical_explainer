from typing import Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

class TokenSHAPModel:
    """
    TokenSHAPModel wraps the language model and tokenizer to facilitate text generation 
    and token-level analysis.
    
    This wrapper is specifically designed to work with the TokenSHAP module for 
    generating explanations for decoder-based language models.
    """
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer, 
                 max_length: int = 256):
        """
        Initialize the TokenSHAP model wrapper.
        
        Args:
            model: Pre-trained decoder model (e.g., GPT, Mistral)
            tokenizer: Associated tokenizer for the model
            max_length: Maximum sequence length for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, 
                input_ids: Union[str, torch.Tensor], 
                attention_mask: torch.Tensor = None) -> str:
        """
        Generate text based on input prompt or tokenized input IDs.
        
        Args:
            input_ids: Input text or pre-tokenized input IDs
            attention_mask: Optional attention mask for pre-tokenized input
            
        Returns:
            Generated text as string
            
        Raises:
            ValueError: If tensor input_ids provided without attention_mask
        """
        try:
            # Clear GPU memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Handle string input
            if isinstance(input_ids, str):
                inputs = self.tokenize_input(input_ids)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            # Handle tensor input
            elif isinstance(input_ids, torch.Tensor):
                if attention_mask is None:
                    raise ValueError("attention_mask required when input_ids is tensor")
                if not isinstance(attention_mask, torch.Tensor):
                    raise TypeError("attention_mask must be a torch.Tensor")

            # Move tensors to model's device
            device = self.model.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate text
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,  # Could make this configurable
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode and return
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            raise

    def tokenize_input(self, text: str) -> dict:
        """
        Tokenize input text with proper padding and attention mask.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dict containing input_ids and attention_mask tensors
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True
        )