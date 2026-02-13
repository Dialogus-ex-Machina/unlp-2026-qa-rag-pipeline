from .hyde_prompt import HydePrompt

PROMPT = """
Напиши текст, який відповідає на запитання.
Запитання: {query}
Текст:""".strip()

class UkrHydePrompt(HydePrompt):
    def __init__(self):
        super().__init__(PROMPT)
