from ..base import PromptTemplate

class DefaultTemplate(PromptTemplate):
    def __init__(self):
        super().__init__()
        self.user_prefix = "### Instruction:\n"
        self.user_suffix = "\n\n"
        self.assistant_prefix = "### Response:\n"
        self.assistant_suffix = "\n\n"