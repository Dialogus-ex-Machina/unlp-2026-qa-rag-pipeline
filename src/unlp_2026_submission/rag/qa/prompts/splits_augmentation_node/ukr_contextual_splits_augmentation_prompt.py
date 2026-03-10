from .splits_augmentation_node import SplitsAugmentationPrompt

PROMPT = """
Твоє завдання: згенерувати стислий семантичний опис для наведеного уривку документа. 
Цей опис використовуватиметься для покращення векторного пошуку, тому він має чітко ідентифікувати головну тему, ключові сутності та специфічні терміни уривку.

Обмеження:
- Обсяг: 3-4 речення.
- Суть: синтезуй унікальні ознаки тексту (концепції, алгоритми, імена), які відрізняють його від інших розділів. Не дублюй сам текст.

<text_section>
{%- for split in document_splits %}
{{ split | trim }}
{%- endfor %}
</text_section>

Проаналізуй <text_section> та видай лише згенерований семантичний опис.
""".strip()


class UkrContextualSplitsAugmentationPrompt(SplitsAugmentationPrompt):
    def __init__(self):
        super().__init__(PROMPT)
