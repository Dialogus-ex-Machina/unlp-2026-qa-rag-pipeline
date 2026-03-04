from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from merlin.rag.index import IndexState

PAGE_DELIMITER_PATTERN = re.compile(r"^===== Page (\d+) =====$")

class DelimitedPageLoadNode:
    """
    Reader for `.txt` files where pages are separated with lines like:

        ===== Page {page_num} =====

    The line with the delimiter belongs to the *end* of the corresponding page.
    Each returned `Document` corresponds to one page and carries metadata:
      - `file_name`: the basename of the source file
      - `page_label`: page number as a string (to mirror PDF reader behaviour)
      - any `extra_info` passed to `load_data`
    """

    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state["filepaths"]

        documents: List[Document] = []

        for filepath in filepaths:
            document_pages = self.load_data(filepath)
            documents.extend(document_pages)

        return {"documents": documents}

    def load_data(
        self,
        file: Union[str, Path],
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        path = Path(file)

        # Read the whole file; pages were already merged and annotated upstream.
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        docs: List[Document] = []
        page_lines: List[str] = []

        for line in lines:
            match = PAGE_DELIMITER_PATTERN.match(line)
            if match:
                # Everything accumulated so far is the content of this page.
                page_text = "\n".join(page_lines).strip()
                if page_text:
                    page_num = match.group(1)
                    metadata: Dict[str, str] = {
                        "page_label": page_num,
                        "source": str(path),
                    }
                    if extra_info:
                        metadata.update(extra_info)

                    docs.append(Document(page_content=page_text, metadata=metadata))

                # Reset for the next page's content.
                page_lines = []
            else:
                page_lines.append(line)

        # If there is trailing content after the last delimiter, attach it to the
        # last page we saw is ambiguous; for now we ignore such a trailing tail.

        return docs

