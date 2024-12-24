from ..base import PromptTemplate

class MistralTemplate(PromptTemplate):
    def __init__(self):
        super().__init__()
        self.user_prefix = "<|im_start|>user\n"
        self.user_suffix = "<|im_end|>\n"
        self.assistant_prefix = "<|im_start|>assistant\n"