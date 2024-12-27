from .context_explainer_base import ContextExplainerPromptTemplate
from .factory import PromptTemplateFactory

from .factory import ModelFamily       

from .models import (
    DefaultTemplate,
    LlamaTemplate,
    GemmaTemplate,
    MistralTemplate
)

class ContextExplainerPromptTemplateFactory(PromptTemplateFactory):
    @staticmethod
    def create_template(model_name: str) -> ContextExplainerPromptTemplate:
        """Create appropriate template for model"""
        model_family = ContextExplainerPromptTemplateFactory.detect_model_family(model_name)
        
        templates = {
            ModelFamily.LLAMA: LlamaTemplate,
            ModelFamily.GEMMA: GemmaTemplate,
            ModelFamily.MISTRAL: MistralTemplate,
            ModelFamily.UNKNOWN: DefaultTemplate
        }
        
        template_class = templates.get(model_family, DefaultTemplate)
        return ContextExplainerPromptTemplate(template_class())