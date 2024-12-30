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
        explanation_max_tokens = 512
    ) -> str:
        print("[DEBUG] Generating prompts with provided case, explanation, and prediction.")
        prompts = self.template.generate_prompt(case, explanation, prediction, add_context=True, explanation_method=TokenValuePairMethod.TOKEN_VAL_PAIR)
        # print(f"[DEBUG] Prompts for which explainers generated: {prompts.keys()}")

        responses = {}
        # print("________________________________________________________________________________________________________________")
        for explanation_type, prompt in prompts.items():
            # print(f"[DEBUG] Generating response for model: ({model}, {explanation_type}) with prompt: {prompt}")
            print(f"[DEBUG] Generating response for model: ({model}, {explanation_type})")
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
            # print(f"[DEBUG] Response received for model ({model},{explanation_type}): {completion.choices[0].message.content}")
            responses[explanation_type] = completion.choices[0].message.content

        # print(f"[DEBUG] Final responses: {responses}")
        return responses