from .base import PromptTemplate
from typing import Dict
import os

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
        # print("[DEBUG] Initializing ContextExplainerPromptTemplate with example_template.")
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
        # print("[DEBUG] Converting token value pairs to string with method:", method)
        if method == TokenValuePairMethod.ORDER_BY_IMPORTANCE:
            # Method 1: Order the words by their importance
            # print("[DEBUG] Using ORDER_BY_IMPORTANCE method for token value pair.")
            ordered_list = sorted(word_importance_list, key=lambda x: x[1], reverse=True)
            words = [word for word, _ in ordered_list]
            # print(f"[DEBUG] Ordered list of words by importance: {words}")
            return "Keywords in Order of importance:\n" + ", ".join(words)
        
        elif method == TokenValuePairMethod.TOP_PERCENTILE:
            # Method 3: Highlight words whose importance is in the top N percentile (e.g., top 20%)
            # print("[DEBUG] Using TOP_PERCENTILE method for token value pair.")
            top_percentile = 0.2
            threshold = sorted([abs(value) for _, value in word_importance_list], reverse=True)[
                int(len(word_importance_list) * top_percentile) - 1
            ]
            # print(f"[DEBUG] Calculated threshold for top percentile: {threshold}")
            top_words = [word for word, value in word_importance_list if abs(value) >= threshold]
            # print(f"[DEBUG] Top words selected: {top_words}")
            return "Most influential words:\n" + ", ".join(top_words)
            
        elif method == TokenValuePairMethod.TOKEN_VAL_PAIR:
            # Method 2: Order the words by their importance and share token value pair in string
            # print("[DEBUG] Using TOKEN_VAL_PAIR method for token value pair.")
            ordered_list = sorted(word_importance_list, key=lambda x: x[1], reverse=True)
            # print(f"[DEBUG] Ordered list of words by importance: {ordered_list}")
            return "# MODEL'S TOKENS in Order of Importance (with importance scores):" + "\n-".join(f"'{word}' : {value:.4f}" for word, value in ordered_list)
        
        else:
            # print("[DEBUG] No valid method selected for token value pair.")
            return ""
    

    
    def format_explanations(self, explanation: Dict, explanation_method:TokenValuePairMethod = TokenValuePairMethod.IGNORE):
        # print("[DEBUG] Formatting explanations with method:", explanation_method)
        all_explanations = {}
        for key, values in explanation.items():
            # print(f"[DEBUG] Processing explanation for key: {key}")
            if key == "lime":
                lime_exp = explanation["lime"]["exp"]
                ret_val = ""
                for label in lime_exp.available_labels():
                    # print(f"[DEBUG] Formatting explanation for label: {label}")
                    ret_val += self.token_value_pair_to_string(lime_exp.as_list(label= label), explanation_method) + ""
                    # ret_val += "\nPredicted Probabilities for each option:" + str(lime_exp.predict_proba)
                all_explanations[key] = ret_val
                # print(f"[DEBUG] Formatted explanation for key 'lime': {ret_val}")
                
            if key == "shap":
                shap_exp = explanation["shap"]["exp"]
                ret_val = ""
                word_value_pairs = [(word, shap_value) for word, shap_value in zip(shap_exp.data[0], shap_exp.values[0])]
                # print(f"[DEBUG] Formatting SHAP explanation with word-value pairs: {word_value_pairs}")
                ret_val += self.token_value_pair_to_string(word_value_pairs, explanation_method) +"\n"
                all_explanations[key] = ret_val
                # print(f"[DEBUG] Formatted explanation for key 'shap': {ret_val}")

            if key == "shapley_values":
                token_shap = explanation["shapley_values"]
                word_value_pairs = []
                for key, value in token_shap.items():
                    parts = key.rsplit("_", 1)
                    word_value_pairs.append((parts[0], value))
                # print(f"[DEBUG] Formatting Shapley values explanation with word-value pairs: {word_value_pairs}")
                ret_val += self.token_value_pair_to_string(word_value_pairs, explanation_method) +"\n"
                all_explanations[key] = ret_val
        print(f"[DEBUG] Formatted explanation keys for all_explanations in format_explanations: {all_explanations.keys()}")

        return all_explanations
            
    def format_context(self, case: Dict) -> str:
        # print("[DEBUG] Formatting context for case.")
        options_text = self._format_options(case['options'])
        formatted_context = (
            f"# QUESTION : \n{case['full_question']}\n\n"
            f"# AVAILABLE OPTIONS : \n{options_text}\n"
        )
        # print(f"[DEBUG] Formatted context: {formatted_context}")
        return formatted_context
    
    def default_prompt(self, user_prefix, context_text, prediction, explanation_text, user_suffix, assistant_prefix)->str:
        return (f" {user_prefix}\n"
                f"You are a Medical Expert. Evaluate the answer given by a model that is trained for answering medical question and answer. Explain why the correct answer is selected. \n\nCLINICAL CASE:\n"
                f"{context_text}" 
                f"CORRECT OPTION: {prediction}"
                f"{explanation_text}\n"
                f"Based on the question, predicted option and the model's token importance scores, explain the diagnosis.\n"
                f"{user_suffix}"
                f"{assistant_prefix}")
        
    def generate_prompt(self, case: Dict, explanation: Dict, prediction:Dict, add_context: bool, custom_prompt = None, explanation_method:TokenValuePairMethod= TokenValuePairMethod.IGNORE) -> Dict:
        """Format prompt with case, explanation, and context"""
        print("[DEBUG][generate_prompt] Prompt generation with case, explanation, and prediction.")
        context_text = self.format_context(case) if add_context else ""
        # print(f"[DEBUG] Context text: {context_text}")
        ret_val ={}
        if not custom_prompt:
            custom_prompt = self.default_prompt
        for model, explanation_text in self.format_explanations(explanation, explanation_method).items():
                ret_val[model] = custom_prompt(self.user_prefix, context_text, prediction['prediction'], explanation_text, self.user_suffix, self.assistant_prefix)
        print(f"[DEBUG] Example of generated prompts for all models: {ret_val[ret_val.keys()[0]]}")
        return ret_val