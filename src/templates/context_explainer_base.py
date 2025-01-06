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
    """
    Prompt template that supports explanation and additional context for use in phase 2 of a pipeline.

    Attributes:
    - user_prefix: String prefix for user inputs.
    - user_suffix: String suffix for user inputs.
    - assistant_prefix: String prefix for assistant responses.
    - assistant_suffix: String suffix for assistant responses.
    """
    def __init__(self, example_template: PromptTemplate):
        """
        Initialize the template with a base prompt template.

        Args:
        - example_template: A base PromptTemplate object to initialize prefixes and suffixes.
        """
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
        """
        Format explanations based on the specified method for token-value pair processing.

        Args:
        - explanation: Dictionary containing explanations from various models.
        - explanation_method: TokenValuePairMethod enum specifying the processing method.

        Returns:
        - A dictionary of formatted explanations for each model.
        """
        all_explanations = {}
        for key, values in explanation.items():
            if key == "lime":
                lime_exp = explanation["lime"]["exp"]
                ret_val = ""
                for label in lime_exp.available_labels():
                    ret_val += self.token_value_pair_to_string(lime_exp.as_list(label= label), explanation_method) + ""
                all_explanations[key] = ret_val
            
                
            if key == "shap":
                shap_exp = explanation["shap"]["exp"]
                ret_val = ""
                word_value_pairs = [(word, shap_value) for word, shap_value in zip(shap_exp.data[0], shap_exp.values[0])]
            
                ret_val += self.token_value_pair_to_string(word_value_pairs, explanation_method) +"\n"
                all_explanations[key] = ret_val
            

            if key == "token_shap":
                token_shap = explanation["token_shap"]["shapley_values"]
                word_value_pairs = []
                ret_val = ""
                for key_token_shap, value_token_shap in token_shap.items():
                    parts = key_token_shap.rsplit("_", 1)
                    word_value_pairs.append((parts[0], value_token_shap))
                ret_val += self.token_value_pair_to_string(word_value_pairs, explanation_method) +"\n"
                all_explanations[key] = ret_val
    
        return all_explanations
            
    def format_context(self, case: Dict) -> str:
        """
        Format a case dictionary into a string with question and options.

        Args:
        - case: Dictionary containing the question and available options.

        Returns:
        - A formatted string representing the case context.
        """
        options_text = self._format_options(case['options'])
        formatted_context = (
            f"# QUESTION : \n{case['full_question']}\n\n"
            f"# AVAILABLE OPTIONS : \n{options_text}\n"
        )
    
        return formatted_context
    
    def default_prompt(self, user_prefix, context_text, prediction, explanation_text, user_suffix, assistant_prefix)->str:
        """
        Generate the default prompt template.

        Args:
        - user_prefix: Prefix for the user input.
        - context_text: Importance of different tokens in text format.
        - prediction: Predicted answer: Either MCQ or the Generated text.
        - explanation_text: Explanation for the prediction.
        - user_suffix: Suffix for the user input.
        - assistant_prefix: Prefix for the assistant response.

        Returns:
        - A formatted string prompt.
        """
        return (f" {user_prefix}\n"
                f"You are a Medical Expert. Evaluate the answer given by a model that is trained for answering medical question and answer. Explain why the correct answer is selected. \n\nCLINICAL CASE:\n"
                f"{context_text}" 
                f"CORRECT OPTION: {prediction}"
                f"{explanation_text}\n"
                f"Based on the question, predicted option and the model's token importance scores, explain the diagnosis.\n"
                f"{user_suffix}"
                f"{assistant_prefix}")
        
    def generate_prompt(self, case: Dict, explanation: Dict, prediction:Dict, add_context: bool, custom_prompt = None, explanation_method:TokenValuePairMethod= TokenValuePairMethod.IGNORE) -> Dict:
        """
        Generate prompts for different models based on the case and explanation.

        Args:
        - case: Dictionary containing case details.
        - explanation: Dictionary of explanations from different models.
        - prediction: Dictionary containing model predictions.
        - add_context: Boolean to include or exclude context in the prompt.
        - custom_prompt: Optional custom prompt function.
        - explanation_method: TokenValuePairMethod enum specifying the explanation method.

        Returns:
        - A dictionary of formatted prompts for each model.
        """
        context_text = self.format_context(case) if add_context else ""
    
        ret_val ={}
        if custom_prompt == None:
            custom_prompt = self.default_prompt
        all_explanations = self.format_explanations(explanation, explanation_method)
    
    
        for model, explanation_text in all_explanations.items():
                if model == "lime" or model == "shap":
                    ret_val[model] = custom_prompt(self.user_prefix, context_text, prediction['prediction'], explanation_text, self.user_suffix, self.assistant_prefix)
                if model == "token_shap":
                    ret_val[model] = custom_prompt(self.user_prefix, context_text, prediction['response'], explanation_text, self.user_suffix, self.assistant_prefix)
    
        return ret_val