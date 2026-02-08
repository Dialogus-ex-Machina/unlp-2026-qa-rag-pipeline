from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


PAGE_DELIMITER_PATTERN = re.compile(r"^===== Page (\d+) =====$")


class DelimitedPageTxtReader(BaseReader):
    """
    Reader for `.txt` files where pages are separated with lines like:

        ===== Page {page_num} =====

    The line with the delimiter belongs to the *end* of the corresponding page.
    Each returned `Document` corresponds to one page and carries metadata:
      - `file_name`: the basename of the source file
      - `page_label`: page number as a string (to mirror PDF reader behaviour)
      - any `extra_info` passed to `load_data`
    """

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
                        "file_name": path.name,
                    }
                    if extra_info:
                        metadata.update(extra_info)

                    docs.append(Document(text=page_text, metadata=metadata))

                # Reset for the next page's content.
                page_lines = []
            else:
                page_lines.append(line)

        # If there is trailing content after the last delimiter, attach it to the
        # last page we saw is ambiguous; for now we ignore such a trailing tail.

        return docs

