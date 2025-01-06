from abc import ABC
from typing import Dict

class PromptTemplate(ABC):
    """Base class for all prompt templates"""
    def __init__(self):
        self.user_prefix = ""
        self.user_suffix = ""
        self.assistant_prefix = ""
        self.assistant_suffix = ""

    def _format_options(self, options: Dict) -> str:
        """Format options dictionary into string"""
        return "\n".join([f"{k}: {v}" for k, v in options.items() if v is not None])

    def format_prompt(self, case: Dict) -> str:
        """Template method to format the entire prompt"""
        options_text = self._format_options(case['options'])
        
        return (
            f"{self.user_prefix}"
            f"Analyze this clinical case and provide your diagnosis:\n\n"
            f"{case['full_question']}\n\n"
            f"Options:\n{options_text}\n"
            f"Select the most appropriate option with concise explanation.\n"
            f"{self.user_suffix}"
            f"{self.assistant_prefix}"
        )