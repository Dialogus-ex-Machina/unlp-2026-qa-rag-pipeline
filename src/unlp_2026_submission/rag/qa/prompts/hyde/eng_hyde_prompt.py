from .hyde_prompt import HydePrompt

# TODO align passage structure and size with chunk
PROMPT= """
Please write a passage to answer the question
Question: {query}
Passage:""".strip()

class EngHydePrompt(HydePrompt):
    def __init__(self):
        super().__init__(PROMPT)
