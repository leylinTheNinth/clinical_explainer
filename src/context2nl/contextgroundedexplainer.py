import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from typing import Dict
import os
from groq import Groq
from ..templates import ContextExplainerPromptTemplateFactory
from ..templates.context_explainer_base import TokenValuePairMethod

class ContextGroundedExplainer:
    """
    A class for generating model responses to explain predictions grounded in contextual information.
    It initializes a GROQ client for accessing external APIs and uses a prompt template to generate 
    responses with explanations based on input cases, predictions, and additional explanations.

    Attributes:
    ----------
    client : Groq
        The API client initialized using the GROQ_API_KEY environment variable.
    template : ContextExplainerPromptTemplate
        The prompt template used for generating explanation prompts.

    Methods:
    -------
    generate_response(
        case: Dict, 
        explanation: Dict,
        prediction: Dict,
                                            # The case, explanation, prediction dictionaries are yielded by Phase 1 of the pipeline.
        model: str = "mixtral-8x7b-32768",  # this parameter can also be set to any of the other models available in the GROQ API.
                                            # While executing both phases together this will be set in the process_dataset call.
        explanation_max_tokens: int = 512,
        custom_prompt: Callable = None      # Custom prompts can be passed to the template for generating responses as directed in the README.
    ) -> str
        Generates responses for explanations using the provided case, prediction, and model settings.
    """
    def __init__(self):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.template = ContextExplainerPromptTemplateFactory.create_template() #Creates a template of default class
        print("Context Grounded Explainer Initialized successfully: ", self.template)
        print("Client successfully set up: ",self.client)


    def generate_response(
        self,
        case: Dict, 
        explanation: Dict,
        prediction: Dict,
        model : str = "mixtral-8x7b-32768",
        explanation_max_tokens = 512,
        custom_prompt = None
    ) -> str:
        prompts = self.template.generate_prompt(case, explanation, prediction, add_context=True, explanation_method=TokenValuePairMethod.TOKEN_VAL_PAIR, custom_prompt=custom_prompt)

        responses = {}
        for explanation_type, prompt in prompts.items():
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=1,
                max_tokens=explanation_max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )
            responses[explanation_type] = completion.choices[0].message.content

        return responses