from .splits_augmentation_node import SplitsAugmentationPrompt

PROMPT = """
Ми хочемо розмістити фрагмент в контексті всього документа.

Документ:
{doc_content}

Фрагмент:
{chunk_content}

Надай короткий і стислий контекст для цього фрагмента в межах усього документа, щоб покращити семантичний векторний пошук цього фрагмента.
Відповідай лише стислим контекстом.
""".strip()

class UkrContextualSplitsAugmentationPrompt(SplitsAugmentationPrompt):
    def __init__(self):
        super().__init__(PROMPT)
