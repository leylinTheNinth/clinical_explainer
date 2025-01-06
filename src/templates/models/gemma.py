from ..base import PromptTemplate

class GemmaTemplate(PromptTemplate):
    def __init__(self):
        super().__init__()
        self.user_prefix = "<start_of_turn>user\n"
        self.user_suffix = "<end_of_turn>\n"
        self.assistant_prefix = "<start_of_turn>assistant\n"