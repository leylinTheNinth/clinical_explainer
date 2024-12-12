
from datasets import load_dataset
from typing import Dict, List, Generator
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

class Pipeline:
    def __init__(self, model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize pipeline for CasiMedicos-Arg dataset.
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device
        
        # Components
        self.tokenizer = None
        self.model = None
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'correct_predictions': 0,
            'errors': []
        }

    def setup(self):
        """Initialize model and tokenizer"""
        try:
            print(f"Loading model and tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMultipleChoice.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            return self
        except Exception as e:
            raise RuntimeError(f"Pipeline setup failed: {str(e)}")

    def _preprocess_generator(self, cases: List[Dict]) -> Generator:
        """Generate preprocessed cases one at a time"""
        for case in cases:
            try:
                # Remove None options
                valid_options = {k: v for k, v in case['options'].items() if v is not None}
                option_texts = list(valid_options.values())
                option_keys = list(valid_options.keys())
                
                # Tokenize
                encodings = self.tokenizer(
                    [case['full_question']] * len(option_texts),
                    option_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Add batch dimension
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

    def _predict_generator(self, processed_cases: Generator) -> Generator:
        """Generate predictions with detailed live output"""
        for case in processed_cases:
            if 'error' in case:
                print(f"\n❌ Error in case {case['original'].get('id', 'unknown')}: {case['error']}")
                yield case
                continue
                
            try:
                print("\n" + "="*80)
                print(f"📋 Case ID: {case['original'].get('id', 'unknown')}")
                print(f"Type: {case['original']['type']}")
                
                # Print full question
                print("\n📝 Question:")
                print(case['original']['full_question'])
                
                # Print all options
                print("\n🔤 Options:")
                valid_options = {k: v for k, v in case['original']['options'].items() if v is not None}
                for key, text in valid_options.items():
                    print(f"Option {key}: {text}")
                
                # Make prediction
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
                    'is_correct': is_correct
                }
                
                yield result
                    
            except Exception as e:
                print(f"\n❌ Error predicting case {case['original'].get('id', 'unknown')}: {str(e)}")
                yield {'error': str(e), 'original': case['original']}

    def process_dataset(self, split: str = 'validation', limit: int = None) -> List[Dict]:
        """
        Process CasiMedicos dataset.
        Args:
            split: 'train', 'validation', or 'test'
            limit: number of cases to process (None for all cases)
        """
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
