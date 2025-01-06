from ..base import PromptTemplate

class LlamaTemplate(PromptTemplate):
    def __init__(self):
        super().__init__()
        self.user_prefix = "<s>[INST] "
        self.user_suffix = " [/INST]"
        self.assistant_prefix = "\n"