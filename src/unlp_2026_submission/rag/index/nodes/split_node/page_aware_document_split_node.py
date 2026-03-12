import copy
import re
from re import Pattern
from typing import List, Optional, Tuple, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

from unlp_2026_submission.rag.index.index_state import IndexState


class PageAwareDocumentSplitNode:
    def __init__(
            self,
            splitter: BaseDocumentTransformer,
            page_delimiter: Pattern[str] | None = None,
            min_fuzzy_score: float = 95.0,
    ):
        self.splitter = splitter
        self.min_fuzzy_score = min_fuzzy_score

        if page_delimiter is None:
            self.page_delimiter = re.compile(r"===== Page (\d+) =====")

    def __call__(self, state: IndexState) -> IndexState:
        documents = state["documents"]
        result_splits = []

        for document in documents:
            raw_text = document.page_content or ""

            clean_text, page_ranges = self._get_clean_text_with_pages_range(raw_text)
            doc_metadata = copy.deepcopy(document.metadata)
            print('Start document splitting')
            raw_splits = self.splitter.transform_documents([
                Document(page_content=clean_text, metadata=doc_metadata)
            ])
            print('End document splitting. Docs:', len(raw_splits))

            document_splits = self._add_page_metadata_to_splits(
                clean_text=clean_text,
                splits=raw_splits,
                page_ranges=page_ranges,
            )
            print('End page metadata adding')
            result_splits.extend(document_splits)

        return { "splits": result_splits }

    def _add_page_metadata_to_splits(
            self,
            clean_text: str,
            splits: Sequence[Document],
            page_ranges: List[Tuple[int, int, int]],
    ) -> Sequence[Document]:
        if not page_ranges:
            for s in splits:
                s.metadata["page_label"] = 1
            return splits

        last_page_num = page_ranges[0][2]

        for s in splits:
            start = s.metadata.get("start_index")

            if (
                    start is None
                    or not isinstance(start, int)
                    or start < 0
            ):
                s.metadata["page_label"] = last_page_num
                continue

            end = s.metadata.get("end_index") or s.metadata.get("start_index") + len(s.page_content)

            # Fallback if missing/invalid
            if (
                    end is None
                    or not isinstance(end, int)
                    or end < 0
                    or start >= end
            ):
                s.metadata["page_label"] = last_page_num
                continue

            # Clamp to clean_text bounds (safety)
            start = max(0, min(start, len(clean_text)))
            end = max(0, min(end, len(clean_text)))
            if start >= end:
                s.metadata["page_label"] = last_page_num
                continue

            page_num = self._dominant_page_for_span(start, end, page_ranges)
            s.metadata["page_label"] = page_num
            last_page_num = page_num

        return splits

    def _get_clean_text_with_pages_range(
            self,
            raw: str
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Reads raw text where the delimiter "===== Page N =====" appears at the END of page N.

        Returns:
          - clean_text: full text with delimiters removed (pages joined by "\\n\\n")
          - page_ranges: list of (start_char, end_char, page_number) spans in clean_text
        """
        matches = list(self.page_delimiter.finditer(raw))
        if not matches:
            clean = raw.strip()
            return clean, [(0, len(clean), 1)]

        pages: List[Tuple[int, str]] = []
        last_end = 0
        last_page_num: Optional[int] = None

        for m in matches:
            page_num = int(m.group(1))
            page_text = raw[last_end: m.start()].strip()
            if page_text:
                pages.append((page_num, page_text))
                last_page_num = page_num
            last_end = m.end()

        # If there is trailing text after the last delimiter, assign it to next page number.
        tail = raw[last_end:].strip()
        if tail:
            next_page = (last_page_num + 1) if last_page_num is not None else 1
            pages.append((next_page, tail))

        sep = "\n\n"
        clean_parts = [t for (_, t) in pages]
        clean_text = sep.join(clean_parts)

        page_ranges: List[Tuple[int, int, int]] = []
        offset = 0
        for (page_num, page_text) in pages:
            start = offset
            end = offset + len(page_text)
            page_ranges.append((start, end, page_num))
            offset = end + len(sep)

        return clean_text, page_ranges

    # -----------------------------
    # Step 3: Page dominance by overlap
    # -----------------------------
    def _dominant_page_for_span(
            self,
            span_start: int,
            span_end: int,
            page_ranges: List[Tuple[int, int, int]],
    ) -> int:
        """
        Given a [span_start, span_end) in clean_text, return the page_number that overlaps
        the most characters.
        """
        if not page_ranges:
            return 1

        best_page = page_ranges[0][2]
        best_overlap = -1

        for (p_start, p_end, page_num) in page_ranges:
            overlap_start = max(span_start, p_start)
            overlap_end = min(span_end, p_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_num

        return best_page
