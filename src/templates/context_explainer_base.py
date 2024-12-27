from .base import PromptTemplate
from typing import Dict
import numpy as np

from enum import Enum
from typing import List, Tuple
class TokenValuePairMethod(Enum):
    IGNORE = 0
    ORDER_BY_IMPORTANCE = 1
    TOP_PERCENTILE = 2
    TOKEN_VAL_PAIR = 3
    
class ContextExplainerPromptTemplate(PromptTemplate):
    """Prompt template that supports explanation and additional context"""

    def __init__(self, example_template: PromptTemplate=TokenValuePairMethod.IGNORE):
        self.user_prefix = example_template.user_prefix
        self.user_suffix = example_template.user_suffix
        self.assistant_prefix = example_template.assistant_prefix
        self.assistant_suffix = example_template.assistant_suffix

    def token_value_pair_to_string(self,
        word_importance_list: List[Tuple[str, float]],
        method: TokenValuePairMethod
    ) -> str:
        """
        Convert a list of (word, importance_value) tuples into a prompt string.
    
        Args:
        - word_importance_list: List of tuples, each containing a word and its importance value.
        - method: An integer indicating the method to use.
    
        Returns:
        - A formatted string prompt.
        """
        if method == TokenValuePairMethod.ORDER_BY_IMPORTANCE:
            # Method 1: Order the words by their importance
            ordered_list = sorted(word_importance_list, key=lambda x: x[1], reverse=True)
            words = [word for word, _ in ordered_list]
            return "Keywords in Order of importance:\n" + ", ".join(words)
        
        elif method == TokenValuePairMethod.TOP_PERCENTILE:
            # Method 3: Highlight words whose importance is in the top N percentile (e.g., top 20%)
            top_percentile = 0.2
            threshold = sorted([abs(value) for _, value in word_importance_list], reverse=True)[
                int(len(word_importance_list) * top_percentile) - 1
            ]
            top_words = [word for word, value in word_importance_list if abs(value) >= threshold]
            return "Most influential words:\n" + ", ".join(top_words)
            
        elif method == TokenValuePairMethod.TOKEN_VAL_PAIR:
            # Method 2: Order the words by their importance and share token value pair in string
            ordered_list = sorted(word_importance_list, key=lambda x: x[1], reverse=True)
            return "# MODEL'S TOKENS in Order of Importance (with importance scores):" + "\n-".join(f"'{word}' : {value:.4f}" for word, value in ordered_list)
        
        else:
            return ""
    

    
    def format_explanations(self, explanation: Dict, explanation_method:TokenValuePairMethod = TokenValuePairMethod.IGNORE):
        all_explanations = []
        for key, values in explanation.items():
            if key == "lime":
                lime_exp = explanation["lime"]["exp"]
                ret_val = ""
                for label in lime_exp.available_labels():
                    ret_val += self.token_value_pair_to_string(lime_exp.as_list(label= label), explanation_method) + ""
                    # ret_val += "\nPredicted Probabilities for each option:" + str(lime_exp.predict_proba)
                all_explanations.append(ret_val)
                
            if key == "shap":
                shap_exp = explanation["shap"]["exp"]
                ret_val = ""
                word_values_pairs = [(word, shap_value) for word, shap_value in zip(shap_exp.data[0], shap_exp.values[0])]
                ret_val += self.token_value_pair_to_string(word_values_pairs, explanation_method) +"\n"
                all_explanations.append(ret_val)

            if key == "shapley_values":
                token_shap = explanation["shapley_values"]
                word_value_pairs = []
                for key, value in token_shap.items():
                    parts = key.rsplit("_", 1)
                    word_value_pairs.append((parts[0], value))
                ret_val += self.token_value_pair_to_string(word_values_pairs, explanation_method) +"\n"
                all_explanations.append(ret_val)
        return all_explanations
            
    def format_context(self, case: Dict) -> str:
        options_text = self._format_options(case['options'])
        return (
            f"# QUESTION : \n{case['full_question']}\n\n"
            f"# AVAILABLE OPTIONS : \n{options_text}\n"
        )
        
    def generate_prompt(self, case: Dict, explanation: Dict, prediction:Dict, add_context: bool, explanation_method:TokenValuePairMethod= TokenValuePairMethod.IGNORE) -> str:
        """Format prompt with case, explanation, and context"""
        explanation_texts = self.format_explanations(explanation, explanation_method)  # Assuming you have a method for formatting explanation
    
        # Use add_context to decide whether to include the context
        context_text = self.format_context(case) if add_context else ""
    
        return [
            (f" {self.user_prefix}\n"
             f"You are a Medical Expert. Evaluate the answer given by a model that is trained for answering medical question and answer. Explain why the correct answer selected is \n\nCLINICAL CASE:\n"
            f"{context_text}" 
            f"CORRECT OPTION: {prediction['prediction']}"
            f"{explanation_text}\n"
            f"Based on the question, predicted option and the model's token importance scores, explain the diagnosis"
            f"{self.user_suffix}"
            f"{self.assistant_prefix} ") for explanation_text in explanation_texts
        ]
