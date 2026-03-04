from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from merlin.rag.index.index_state import IndexState


class DoclingMarkdownPageSplitNode:
    def __init__(
        self,
        md_suffix: str = ".md",
        marker_regex: str = r"<!--\s*page_end:(-?\d+)\s*-->",
        encoding: str = "utf-8",
        keep_empty_pages: bool = False,
        prefer_pdf_source: bool = True,
    ):
        self.md_suffix = md_suffix
        self.marker_re = re.compile(marker_regex)
        self.encoding = encoding
        self.keep_empty_pages = keep_empty_pages
        self.prefer_pdf_source = prefer_pdf_source

    def __call__(self, state: IndexState) -> IndexState:
        md_paths = self._collect_md_paths(state)

        splits: List[Document] = []
        for md_path in md_paths:
            md_text = Path(md_path).read_text(encoding=self.encoding)

            pdf_source = self._infer_source(md_path)

            pages = self._split_markdown_into_pages(md_text)
            for page_no, page_text in pages:
                page_text = page_text.strip()
                if not page_text and not self.keep_empty_pages:
                    continue

                splits.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf_source,
                            "page": page_no,
                            "md_path": md_path,
                        },
                    )
                )

        splits.sort(key=lambda d: (str(d.metadata.get("source", "")), int(d.metadata.get("page", -1))))
        return {"splits": splits}

    def _collect_md_paths(self, state: IndexState) -> List[str]:
        paths: List[str] = []

        docs = state.get("documents") or []
        for d in docs:
            md_path = (d.metadata or {}).get("md_path")
            if md_path and str(md_path).lower().endswith(".md"):
                paths.append(str(md_path))

        if paths:
            return paths

        filepaths = state.get("filepaths") or []
        for fp in filepaths:
            p = Path(fp)
            if p.suffix.lower() == ".md":
                paths.append(str(p))
            else:
                paths.append(str(p.with_suffix(self.md_suffix)))

        return paths

    def _infer_source(self, md_path: str) -> str:
        if not self.prefer_pdf_source:
            return md_path

        p = Path(md_path)
        pdf_candidate = p.with_suffix(".pdf")
        if pdf_candidate.exists():
            return str(pdf_candidate)
        return str(p)

    def _split_markdown_into_pages(self, md_text: str) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []

        pos = 0
        last_page: Optional[int] = None

        for m in self.marker_re.finditer(md_text):
            page_text = md_text[pos : m.start()]
            page_no = int(m.group(1))
            out.append((page_no, page_text))
            pos = m.end()
            last_page = page_no

        tail = md_text[pos:].strip()
        if tail:
            fallback_page = (last_page + 1) if last_page is not None else 0
            out.append((fallback_page, tail))

        return out