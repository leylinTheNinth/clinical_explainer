from enum import Enum, auto
from .base import PromptTemplate
from .models import (
    DefaultTemplate,
    LlamaTemplate,
    GemmaTemplate,
    MistralTemplate
)

class ModelFamily(Enum):
    LLAMA = auto()
    GEMMA = auto()
    MISTRAL = auto()
    UNKNOWN = auto()

class PromptTemplateFactory:
    @staticmethod
    def detect_model_family(model_name: str) -> ModelFamily:
        """Detect model family from model name"""
        model_name = model_name.lower()
        if "llama" in model_name:
            return ModelFamily.LLAMA
        elif "gemma" in model_name:
            return ModelFamily.GEMMA
        elif "mistral" in model_name:
            return ModelFamily.MISTRAL
        return ModelFamily.UNKNOWN

    @staticmethod
    def create_template(model_name: str) -> PromptTemplate:
        """Create appropriate template for model"""
        model_family = PromptTemplateFactory.detect_model_family(model_name)
        
        templates = {
            ModelFamily.LLAMA: LlamaTemplate,
            ModelFamily.GEMMA: GemmaTemplate,
            ModelFamily.MISTRAL: MistralTemplate,
            ModelFamily.UNKNOWN: DefaultTemplate
        }
        
        template_class = templates.get(model_family, DefaultTemplate)
        return template_class()