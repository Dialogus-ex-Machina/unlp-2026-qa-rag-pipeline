from typing import List, Optional, Dict, Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, ThreadedPdfPipelineOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

from unlp_2026_submission.rag.index.index_state import IndexState


class DoclingLoadSplitNode:
    def __init__(
        self,
        embeddings: Embeddings,
        max_tokens: int = 450,
        merge_peers: bool = True,
        zero_based_pages: bool = True,
        device: str = "auto",
        threaded: bool = True,
        do_ocr: bool = False,
        ocr_backend: str = "torch",
        layout_batch_size: int = 64,
        ocr_batch_size: int = 64,
        table_batch_size: int = 4,
        warmup: bool = False,
    ):
        self.embeddings = embeddings
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.zero_based_pages = zero_based_pages

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
            return {"splits": []}

        loader = DoclingLoader(
            file_path=filepaths,
            export_type=ExportType.DOC_CHUNKS,
            chunker=self._chunker,
            converter=self._converter
        )
        raw_chunks: List[Document] = loader.load()

        splits = [self._normalize_chunk(raw_chunk) for raw_chunk in raw_chunks]

        return {"splits": splits}

    @staticmethod
    def _build_converter(
        device: str,
        threaded: bool,
        do_ocr: bool,
        ocr_backend: str,
        layout_batch_size: int,
        ocr_batch_size: int,
        table_batch_size: int,
        warmup: bool,
    ):
        pipeline_cls = None
        if threaded:
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
            pipeline_cls = ThreadedStandardPdfPipeline

        if threaded:
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=device),
                layout_batch_size=layout_batch_size,
                ocr_batch_size=ocr_batch_size,
                table_batch_size=table_batch_size,
            )
        else:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(device=device)

        pipeline_options.do_ocr = do_ocr
        if do_ocr:
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

    def _build_chunker_from_embeddings(self, embeddings: Embeddings):
        client = getattr(embeddings, "client", None)
        tokenizer = getattr(client, "tokenizer", None) if client is not None else None

        if tokenizer is None:
            model_name = getattr(embeddings, "model_name", None)
            if model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer is None:
            raise ValueError(
                "Could not infer a HuggingFace tokenizer from the provided embeddings instance. "
                "Expected embeddings.client.tokenizer or embeddings.model_name to exist."
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

        new_meta: Dict[str, Any] = {"source": source}
        if page != -1:
            new_meta["page"] = page

        return Document(page_content=doc.page_content, metadata=new_meta)

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
