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
        print("Context Grounded Explainer Initialized successfully")


    def generate_response(
        self,
        case: Dict, 
        explanation: Dict,
        prediction: Dict,
        model : str = "mixtral-8x7b-32768",
        explanation_max_tokens = 512
    ) -> str:
        prompts = self.template.generate_prompt(case, explanation, prediction, add_context = True, explanation_method=TokenValuePairMethod.TOKEN_VAL_PAIR)
        responses = []
        for prompt in prompts:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=1,
                max_tokens= explanation_max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )

            responses.append(completion.choices[0].message.content)

        return responses

    # def __init__(self):
    #     """
    #     Initialize BioMistral model and tokenizer with quantization configuration.
    #     """
    #     print("[DEBUG] Starting ContextGroundedExplainer Initialization.")
    #     self.model_name = "BioMistral/BioMistral-7B"
    #     # self.quantization_config = BitsAndBytesConfig(
    #     #     load_in_8bit=True,
    #     #     llm_int8_threshold=6.0,
    #     #     llm_int8_has_fp16_weight=False,
    #     #     llm_int8_enable_fp32_cpu_offload=True
    #     # )
    #     self.quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16
    #     )

    #     print("[DEBUG] Quantization Initialized")
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, return_tensors="pt", padding=True, truncation=True, add_eos_token=True, add_bos_token=True)
    #     print("[DEBUG] Tokenizer Initialized")
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         quantization_config=self.quantization_config,
    #         device_map="auto",
    #         # torch_dtype=torch.float16
    #     )
    #     print("[DEBUG] Model Initialized")

    #     # Initialize the prompt template for BioMistral
    #     self.template = ContextExplainerPromptTemplateFactory.create_template("mistral")
    #     print("Context Grounded Explainer Initialized successfully")

    # def __init__(self):
    #     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # Model and authentication details
    #     model_name = "google/gemma-2-2b-it"  # Try switching to a smaller model if needed

    #     # Load tokenizer and model
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForCausalLM.from_pretrained(model_name)

    #     # Move model to GPU
    #     self.model = self.model.to(device)

    #     # Use mixed precision for faster processing and lower memory usage
    #     # Load model and tokenizer in FP16 precision if supported
    #     self.model = self.model.half()  # This enables FP16 precision for model weights

    #     self.template = ContextExplainerPromptTemplateFactory.create_template("gemma")
    #     print("Context Grounded Explainer Initialized successfully")


    # def generate_response(
    #     self, 
    #     case: dict, 
    #     explanation: dict,
    # ) -> str:
    #     """
    #     Generate a response from the BioMistral model for a given case.
    #     """
    #     print(f"[DEBUG] In generate response.\ncase:{case}\nexplanation:{explanation}")
    #     print(f"[DEBUG] Explantation type: {type(explanation)}")
    #     # Format the case into a prompt
    #     prompts = self.template.generate_prompt(case, explanation, add_context = True, explanation_method=TokenValuePairMethod.TOKEN_VAL_PAIR)

    #     response = []

    #     self.model.eval()

    #     # Tokenize and generate a response
    #     for prompt in prompts:
    #         print("Processing context grounded prompt: \n", prompt)
    #         inputs = self.tokenizer(prompt, return_tensors="pt", padding = True, truncation = True, max_length = 512).to(self.model.device)
    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens= 512,
    #                 ad_token_id=self.tokenizer.pad_token_id
    #             )
    #         response.append[self.tokenizer.decode(outputs[0], skip_special_tokens=True)]

    #     return response

    # def generate_response(
    #     self, 
    #     case: Dict, 
    #     explanation: Dict,
    #     prediction: Dict
    # ) -> str:
    #     """
    #     Generate a response from the BioMistral model for a given case.
    #     """
    #     print(f"[DEBUG] In generate response.\ncase:{case}\nexplanation:{explanation}")
    #     print(f"[DEBUG] Explantation type: {type(explanation)}")
    #     # Format the case into a prompt
    #     prompts = self.template.generate_prompt(case, explanation, prediction, add_context = True, explanation_method=TokenValuePairMethod.TOKEN_VAL_PAIR)
    #     batch_size = 2
    #     responses = []
    #     for i in range(0, len(prompts), batch_size):
    #         batch_prompts = prompts[i:i + batch_size]

    #         # Tokenize the current batch
    #         inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

    #         # Generate responses for the current batch using mixed precision
    #         outputs = self.model.generate(
    #             inputs["input_ids"],
    #             max_new_tokens=1024,
    #             do_sample=True,
    #             temperature=0.7,
    #             repetition_penalty=1.2,
    #             top_k=50,
    #             top_p=0.95
    #         )

    #         # Decode and clean up prompt repetitions
    #         batch_responses = [
    #             self.tokenizer.decode(output, skip_special_tokens=True).replace(prompt, "").strip()
    #             for output, prompt in zip(outputs, batch_prompts)
    #         ]
    #         responses.extend(batch_responses)

    #         # Clear GPU memory after each batch to prevent memory overflow
    #         torch.cuda.empty_cache()
    #     for i, (prompt, response) in enumerate(zip(prompts, responses)):
    #         print("_________________________________________________________________________________________________")
    #         print(f"Prompt {i + 1}: \n{prompt}")
    #         print(f"Response {i + 1}: \n{response}\n")

    #     return responses