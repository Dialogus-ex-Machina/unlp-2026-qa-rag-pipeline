from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

from unlp_2026_submission.rag.index.index_state import IndexState


class DoclingPageLoadNode:
    def __init__(
        self,
        embeddings: Embeddings,
        max_tokens: int = 450,
        merge_peers: bool = True,
        zero_based_pages: bool = False,
        join_sep: str = "\n\n",
        strip_repeated_headings: bool = True,
        device: str = "auto",
        threaded: bool = True,
        do_ocr: bool = False,
        ocr_backend: str = "torch",
        layout_batch_size: int = 2,
        ocr_batch_size: int = 2,
        table_batch_size: int = 1,
        warmup: bool = False,
    ):
        self.embeddings = embeddings
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.zero_based_pages = zero_based_pages
        self.join_sep = join_sep
        self.strip_repeated_headings = strip_repeated_headings

        self._chunker = self._build_chunker_from_embeddings(embeddings)
        self._converter = self._build_converter(
            device=device,
            threaded=threaded,
            do_ocr=do_ocr,
            ocr_backend=ocr_backend,
            layout_batch_size=layout_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
            warmup=warmup,
        )

    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state["filepaths"]
        if not filepaths:
            return {"documents": []}

        loader = DoclingLoader(
            file_path=filepaths,
            export_type=ExportType.DOC_CHUNKS,
            chunker=self._chunker,
            converter=self._converter,
        )
        raw_chunks: List[Document] = loader.load()

        normalized = [self._normalize_chunk(c) for c in raw_chunks]
        pages = self._group_chunks_to_pages(normalized)

        return {"documents": pages}

    def _build_converter(
        self,
        device: str,
        threaded: bool,
        do_ocr: bool,
        ocr_backend: str,
        layout_batch_size: int,
        ocr_batch_size: int,
        table_batch_size: int,
        warmup: bool,
    ):
        device_value: Any = device
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice

            d = device.lower()
            if d.startswith("cuda"):
                device_value = AcceleratorDevice.CUDA
            elif d == "cpu":
                device_value = AcceleratorDevice.CPU
            elif d == "mps":
                device_value = AcceleratorDevice.MPS
            elif d == "xpu":
                device_value = AcceleratorDevice.XPU
        except Exception:
            pass

        pipeline_cls = None
        if threaded:
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

            pipeline_cls = ThreadedStandardPdfPipeline
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=device_value),
                layout_batch_size=layout_batch_size,
                ocr_batch_size=ocr_batch_size,
                table_batch_size=table_batch_size,
            )
        else:
            pipeline_options = PdfPipelineOptions(accelerator_options=AcceleratorOptions(device=device_value))

        pipeline_options.do_ocr = do_ocr
        if do_ocr:
            from docling.datamodel.pipeline_options import RapidOcrOptions

            pipeline_options.ocr_options = RapidOcrOptions(backend=ocr_backend)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            }
        )

        if warmup:
            converter.initialize_pipeline(InputFormat.PDF)

        return converter

    def _build_chunker_from_embeddings(self, embeddings: Embeddings) -> HybridChunker:
        client = getattr(embeddings, "client", None)
        tokenizer = getattr(client, "tokenizer", None) if client is not None else None

        if tokenizer is None:
            model_name = getattr(embeddings, "model_name", None)
            if model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer is None:
            raise ValueError(
                "Could not infer a HuggingFace tokenizer from embeddings. "
                "Expected embeddings.client.tokenizer or embeddings.model_name."
            )

        return HybridChunker(
            tokenizer=tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=self.merge_peers,
        )

    def _normalize_chunk(self, doc: Document) -> Document:
        md = doc.metadata or {}
        source = md.get("source") or md.get("file_path") or md.get("path") or "unknown"

        dl_meta = md.get("dl_meta") or {}
        page_no = self._extract_page_no(dl_meta)
        page = self._to_page_index(page_no)

        new_meta: Dict[str, Any] = {"source": source, "page": page}
        if page != -1:
            new_meta["page_label"] = str(page)

        headings = dl_meta.get("headings")
        if isinstance(headings, list) and headings:
            new_meta["headings"] = headings

        return Document(page_content=doc.page_content, metadata=new_meta)

    def _group_chunks_to_pages(self, chunks: List[Document]) -> List[Document]:
        buckets: DefaultDict[Tuple[str, int], List[Document]] = defaultdict(list)

        for c in chunks:
            src = (c.metadata or {}).get("source", "unknown")
            page = int((c.metadata or {}).get("page", -1))
            buckets[(src, page)].append(c)

        page_docs: List[Document] = []
        for (src, page), parts in buckets.items():
            texts: List[str] = []
            seen_headings: set[str] = set()

            for p in parts:
                t = p.page_content or ""
                if not t.strip():
                    continue

                if self.strip_repeated_headings:
                    t = self._strip_repeated_heading_prefix(
                        text=t,
                        headings=(p.metadata or {}).get("headings"),
                        seen=seen_headings,
                    )

                t = self._normalize_whitespace(t)
                if t:
                    texts.append(t)

            page_text = self.join_sep.join(texts).strip()
            meta = {"source": src, "page": page, "page_label": str(page)}
            page_docs.append(Document(page_content=page_text, metadata=meta))

        page_docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", -1)))
        return page_docs

    @staticmethod
    def _strip_repeated_heading_prefix(text: str, headings: Any, seen: set[str]) -> str:
        if not isinstance(headings, list) or not headings:
            return text

        out = text

        for h in headings:
            if not isinstance(h, str) or not h.strip():
                continue

            h_norm = h.strip()
            prefix_pattern = r"^\s*" + re.escape(h_norm) + r"\s*\n+"

            if re.match(prefix_pattern, out):
                if h_norm in seen:
                    out = re.sub(prefix_pattern, "", out, count=1)
                else:
                    seen.add(h_norm)

        return out

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _to_page_index(self, page_no: Optional[int]) -> int:
        if page_no is None:
            return -1
        return page_no - 1 if self.zero_based_pages else page_no

    @staticmethod
    def _extract_page_no(dl_meta: Dict[str, Any]) -> Optional[int]:
        items = dl_meta.get("doc_items") or []
        page_nos: List[int] = []

        for it in items:
            for prov in (it.get("prov") or []):
                pn = prov.get("page_no")
                if pn is None:
                    continue
                try:
                    page_nos.append(int(pn))
                except Exception:
                    continue

        return min(page_nos) if page_nos else None
