from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

PROMPT = """
Ти — експерт із сумаризації документів.

Надай коротке резюме наведеного тексту документа, довжиною не більше ніж {char_length} символів. 
Зосередься на вилученні найважливіших сутностей, основної мети та ключових тем. 
Резюме має бути лаконічним і оптимізованим для надання контексту меншим фрагментам тексту. 
Виведи лише текст резюме.

Документ:
{document_content}
"""

class UkrSummarySplitsAugmentationPrompt:
    _template: PromptTemplate

    def __init__(
            self,
    ):
        self._template = PromptTemplate(
            template=PROMPT,
            input_variables=["document_content", "char_length"],
        )

    def format(
            self,
            document_splits: list[Document],
            char_length: int,
            window_size: int,
    ) -> str:
        document_content = self._prepare_document_content(
            document_splits=document_splits,
            window_size=window_size,
        )

        return self._template.format(
            document_content=document_content,
            char_length=char_length,
        )

    def _prepare_document_content(
            self,
            document_splits: list[Document],
            window_size: int,
    ) -> str:
        def _page_label(doc: Document, fallback_idx: int) -> str:
            md = doc.metadata or {}
            for key in ("page_label", "page"):
                if key in md and md[key] is not None:
                    return str(md[key])
            return str(fallback_idx + 1)

        def _render_doc(doc: Document, idx: int) -> str:
            page = _page_label(doc, idx)
            content = (doc.page_content or "").strip()
            return f"[Сторінка {page}]\n{content}"

        n = len(document_splits)
        if n == 0:
            return ""

        window_size = max(0, int(window_size))
        if window_size <= 0 or n <= window_size:
            return "\n\n---\n\n".join(_render_doc(d, i) for i, d in enumerate(document_splits))

        first_k = window_size // 2
        last_k = window_size - first_k

        head = document_splits[:first_k]
        tail = document_splits[-last_k:] if last_k > 0 else []
        omitted = n - (len(head) + len(tail))

        # Avoid inserting a placeholder if it wouldn't separate anything meaningful.
        # (e.g. window_size=1 => only tail; window_size=2 => 1+1 still ok)
        placeholder = (
            "\n[... ПРОПУЩЕНО ЧАСТИНУ ДОКУМЕНТА ...]\n"
            f"(між початком і кінцем є ще {omitted} сторінок. "
            "Під час резюмування врахуй, що середина документа відсутня у вхідному тексті.\n"
        )

        parts: list[str] = []
        parts.extend(_render_doc(d, i) for i, d in enumerate(head))
        if head and tail:
            parts.append(placeholder)
        tail_offset = n - len(tail)
        parts.extend(_render_doc(d, tail_offset + i) for i, d in enumerate(tail))

        return "\n---\n".join(parts)
