from .splits_augmentation_node import SplitsAugmentationPrompt

PROMPT = """
Ось фрагмент, який ми хочемо розмістити в контексті всього документа:
{chunk_content}

Документ:
{doc_content}

Надай короткий і стислий контекст для цього фрагмента в межах усього документа, щоб покращити семантичний векторний пошук цього фрагмента.
Відповідай лише стислим контекстом.
""".strip()

class UkrContextualSplitsAugmentationPrompt(SplitsAugmentationPrompt):
    def __init__(self):
        super().__init__(PROMPT)
