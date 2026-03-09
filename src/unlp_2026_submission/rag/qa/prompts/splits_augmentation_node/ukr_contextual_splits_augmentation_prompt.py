from langchain_core.documents import Document

from .splits_augmentation_node import SplitsAugmentationPrompt

PROMPT = """
Твоє завдання: пояснити семантичний контекст вказаного фрагмента в межах наданого документа. 
Контекст потрібен для покращення векторного пошуку фрагмента.

Обмеження:
- Обсяг: 2-3 речення.
- Не дублюй текст фрагмента, лише пояснюй його суть.
- Формат: звертайся до тексту просто як до "фрагмента". У відповіді не повинно бути жодних цифр, ідентифікаторів чи згадок ID.

<document>
{%- for fragment in fragments %}
<fragment id="{{ fragment.id }}">
{{ fragment.content | trim }}
</fragment>
{%- endfor %}
</document>

Проаналізуй <document> та надай контекст для <fragment id="{{ target_fragment_id }}">.
""".strip()


class UkrContextualSplitsAugmentationPrompt(SplitsAugmentationPrompt):
    def __init__(self):
        super().__init__(PROMPT)

    def format(
        self,
        document_splits: list[Document],
        target_split: Document,
    ) -> str:
        fragments = [
            {"id": i + 1, "content": doc.page_content}
            for i, doc in enumerate(document_splits)
        ]
        target_idx = next(
            i for i, doc in enumerate(document_splits)
            if doc is target_split or doc.page_content == target_split.page_content
        )
        target_fragment_id = target_idx + 1

        return self._template.format(
            fragments=fragments,
            target_fragment_id=target_fragment_id,
        )
