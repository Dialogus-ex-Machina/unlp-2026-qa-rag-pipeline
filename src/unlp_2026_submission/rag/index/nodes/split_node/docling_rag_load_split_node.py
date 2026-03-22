from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

from unlp_2026_submission.rag.index.index_state import IndexState


class DoclingRagLoadSplitNode:
    def __init__(
        self,
        embeddings: Embeddings,
        max_tokens: int = 450,
        merge_peers: bool = True,
        zero_based_pages: bool = False,
        device: str = "auto",
        threaded: bool = True,
        num_threads: int = 4,
        images_scale: float = 2.0,
        table_mode: str = "accurate",
        table_do_cell_matching: bool = False,
        enable_picture_descriptions: bool = False,
        picture_description_mode: str = "smolvlm",
        enable_remote_services: bool = False,
        allow_external_plugins: bool = False,
        ocr_lang: Sequence[str] = ("uk", "en"),
        ocr_engine: str = "easyocr",
        ocr_force_full_page: bool = True,
        ocr_backend: str = "torch",
        layout_batch_size: int = 8,
        ocr_batch_size: int = 8,
        table_batch_size: int = 2,
        selective_ocr: bool = True,
        selective_min_page_chars: int = 120,
        selective_min_alnum_ratio: float = 0.15,
    ):
        self.embeddings = embeddings
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.zero_based_pages = zero_based_pages

        self.device = device
        self.threaded = threaded
        self.num_threads = num_threads

        self.images_scale = images_scale
        self.table_mode = table_mode
        self.table_do_cell_matching = table_do_cell_matching

        self.enable_picture_descriptions = enable_picture_descriptions
        self.picture_description_mode = picture_description_mode
        self.enable_remote_services = enable_remote_services
        self.allow_external_plugins = allow_external_plugins

        self.ocr_lang = tuple(ocr_lang)
        self.ocr_engine = ocr_engine
        self.ocr_force_full_page = ocr_force_full_page
        self.ocr_backend = ocr_backend

        self.layout_batch_size = layout_batch_size
        self.ocr_batch_size = ocr_batch_size
        self.table_batch_size = table_batch_size

        self.selective_ocr = selective_ocr
        self.selective_min_page_chars = selective_min_page_chars
        self.selective_min_alnum_ratio = selective_min_alnum_ratio

        self._chunker = self._build_chunker_from_embeddings(embeddings)

        self._converter_text = self._build_converter(
            device=self.device,
            threaded=self.threaded,
            num_threads=self.num_threads,
            do_ocr=False,
            ocr_engine=self.ocr_engine,
            ocr_lang=self.ocr_lang,
            ocr_force_full_page=False,
            ocr_backend=self.ocr_backend,
            images_scale=self.images_scale,
            table_mode=self.table_mode,
            table_do_cell_matching=self.table_do_cell_matching,
            enable_picture_descriptions=self.enable_picture_descriptions,
            picture_description_mode=self.picture_description_mode,
            enable_remote_services=self.enable_remote_services,
            allow_external_plugins=self.allow_external_plugins,
            layout_batch_size=self.layout_batch_size,
            ocr_batch_size=self.ocr_batch_size,
            table_batch_size=self.table_batch_size,
        )

        self._converter_ocr = self._build_converter(
            device=self.device,
            threaded=self.threaded,
            num_threads=self.num_threads,
            do_ocr=True,
            ocr_engine=self.ocr_engine,
            ocr_lang=self.ocr_lang,
            ocr_force_full_page=self.ocr_force_full_page,
            ocr_backend=self.ocr_backend,
            images_scale=self.images_scale,
            table_mode=self.table_mode,
            table_do_cell_matching=self.table_do_cell_matching,
            enable_picture_descriptions=self.enable_picture_descriptions,
            picture_description_mode=self.picture_description_mode,
            enable_remote_services=self.enable_remote_services,
            allow_external_plugins=self.allow_external_plugins,
            layout_batch_size=self.layout_batch_size,
            ocr_batch_size=self.ocr_batch_size,
            table_batch_size=self.table_batch_size,
        )

    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state.get("filepaths") or []
        if not filepaths:
            return {"splits": []}

        all_out: List[Document] = []

        for fp in filepaths:
            text_chunks = self._load_doc_chunks(fp, converter=self._converter_text)
            text_norm = [self._normalize_chunk(d) for d in text_chunks]

            if not self.selective_ocr:
                all_out.extend(text_norm)
                continue

            bad_pages = self._detect_bad_pages(text_norm)
            if not bad_pages:
                all_out.extend(text_norm)
                continue

            ocr_chunks = self._load_doc_chunks(fp, converter=self._converter_ocr)
            ocr_norm = [self._normalize_chunk(d) for d in ocr_chunks]

            merged = self._merge_pages(text_norm, ocr_norm, bad_pages)
            all_out.extend(merged)

        all_out.sort(key=self._sort_key)
        for d in all_out:
            md = d.metadata or {}
            if "_order" in md:
                md.pop("_order", None)

        return {"splits": all_out}

    def _load_doc_chunks(self, filepath: str, converter: DocumentConverter) -> List[Document]:
        loader = DoclingLoader(
            file_path=[filepath],
            export_type=ExportType.DOC_CHUNKS,
            chunker=self._chunker,
            converter=converter,
        )
        return loader.load()

    def _merge_pages(self, text_chunks: List[Document], ocr_chunks: List[Document], bad_pages: set[int]) -> List[Document]:
        out: List[Document] = []
        by_page_text: DefaultDict[int, List[Document]] = defaultdict(list)
        by_page_ocr: DefaultDict[int, List[Document]] = defaultdict(list)

        for d in text_chunks:
            p = int((d.metadata or {}).get("page", -1))
            by_page_text[p].append(d)

        for d in ocr_chunks:
            p = int((d.metadata or {}).get("page", -1))
            by_page_ocr[p].append(d)

        pages = sorted(set(by_page_text.keys()) | set(by_page_ocr.keys()))
        for p in pages:
            if p in bad_pages and p in by_page_ocr:
                out.extend(by_page_ocr[p])
            else:
                out.extend(by_page_text.get(p, by_page_ocr.get(p, [])))

        return out

    def _detect_bad_pages(self, chunks: List[Document]) -> set[int]:
        page_chars: DefaultDict[int, int] = defaultdict(int)
        page_alnum: DefaultDict[int, int] = defaultdict(int)

        for d in chunks:
            page = int((d.metadata or {}).get("page", -1))
            txt = (d.page_content or "").strip()
            if not txt:
                continue
            page_chars[page] += len(txt)
            page_alnum[page] += sum(1 for ch in txt if ch.isalnum())

        bad: set[int] = set()
        for page, n in page_chars.items():
            if page == -1:
                continue
            ratio = (page_alnum[page] / n) if n else 0.0
            if n < self.selective_min_page_chars or ratio < self.selective_min_alnum_ratio:
                bad.add(page)

        return bad

    def _build_chunker_from_embeddings(self, embeddings: Embeddings) -> HybridChunker:
        client = getattr(embeddings, "client", None)
        tokenizer = getattr(client, "tokenizer", None) if client is not None else None

        if tokenizer is None:
            model_name = getattr(embeddings, "model_name", None)
            if model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer is None:
            raise ValueError("Could not infer a HuggingFace tokenizer from embeddings (need client.tokenizer or model_name).")

        return HybridChunker(
            tokenizer=tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=self.merge_peers,
        )

    @staticmethod
    def _build_converter(
        device: str,
        threaded: bool,
        num_threads: int,
        do_ocr: bool,
        ocr_engine: str,
        ocr_lang: Sequence[str],
        ocr_force_full_page: bool,
        ocr_backend: str,
        images_scale: float,
        table_mode: str,
        table_do_cell_matching: bool,
        enable_picture_descriptions: bool,
        picture_description_mode: str,
        enable_remote_services: bool,
        allow_external_plugins: bool,
        layout_batch_size: int,
        ocr_batch_size: int,
        table_batch_size: int,
    ) -> DocumentConverter:
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        try:
            from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions  # type: ignore
        except Exception:
            ThreadedPdfPipelineOptions = None  # type: ignore

        try:
            from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions  # type: ignore
        except Exception:
            TableFormerMode = None  # type: ignore
            TableStructureOptions = None  # type: ignore

        def _setattr_if(obj: Any, name: str, value: Any) -> None:
            try:
                setattr(obj, name, value)
            except Exception:
                pass

        pipeline_cls = None
        pipeline_options: Any

        if threaded and ThreadedPdfPipelineOptions is not None:
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

            pipeline_cls = ThreadedStandardPdfPipeline
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads),
                layout_batch_size=layout_batch_size,
                ocr_batch_size=ocr_batch_size,
                table_batch_size=table_batch_size,
            )
        else:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=num_threads)

        _setattr_if(pipeline_options, "allow_external_plugins", allow_external_plugins)
        _setattr_if(pipeline_options, "enable_remote_services", enable_remote_services)

        _setattr_if(pipeline_options, "force_backend_text", True)
        _setattr_if(pipeline_options, "images_scale", images_scale)

        _setattr_if(pipeline_options, "do_table_structure", True)
        if TableFormerMode is not None and TableStructureOptions is not None:
            mode = TableFormerMode.ACCURATE if str(table_mode).lower() == "accurate" else TableFormerMode.FAST
            _setattr_if(
                pipeline_options,
                "table_structure_options",
                TableStructureOptions(mode=mode, do_cell_matching=table_do_cell_matching),
            )

        _setattr_if(pipeline_options, "do_ocr", do_ocr)
        if do_ocr:
            ocr_engine_l = str(ocr_engine).lower()
            if ocr_engine_l == "easyocr":
                from docling.datamodel.pipeline_options import EasyOcrOptions

                pipeline_options.ocr_options = EasyOcrOptions(lang=list(ocr_lang), force_full_page_ocr=ocr_force_full_page)
            elif ocr_engine_l in ("rapidocr", "rapid"):
                from docling.datamodel.pipeline_options import RapidOcrOptions

                pipeline_options.ocr_options = RapidOcrOptions(backend=ocr_backend, force_full_page_ocr=ocr_force_full_page)
            elif ocr_engine_l in ("tesseract_cli", "tesseract"):
                try:
                    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

                    pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=ocr_force_full_page)
                except Exception:
                    from docling.datamodel.pipeline_options import TesseractOcrOptions

                    pipeline_options.ocr_options = TesseractOcrOptions(force_full_page_ocr=ocr_force_full_page)

        if enable_picture_descriptions:
            _setattr_if(pipeline_options, "do_picture_description", True)
            pd_mode = str(picture_description_mode).lower()

            if pd_mode == "smolvlm":
                from docling.datamodel.pipeline_options import smolvlm_picture_description

                _setattr_if(pipeline_options, "picture_description_options", smolvlm_picture_description)
            elif pd_mode == "granite":
                from docling.datamodel.pipeline_options import granite_picture_description

                _setattr_if(pipeline_options, "picture_description_options", granite_picture_description)
            else:
                try:
                    from docling.datamodel.pipeline_options import smolvlm_picture_description

                    _setattr_if(pipeline_options, "picture_description_options", smolvlm_picture_description)
                except Exception:
                    pass
        else:
            _setattr_if(pipeline_options, "do_picture_description", False)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            }
        )
        return converter

    def _normalize_chunk(self, doc: Document) -> Document:
        md = doc.metadata or {}
        source = md.get("source") or md.get("file_path") or md.get("path") or "unknown"

        dl_meta = md.get("dl_meta") or {}
        page_no = self._extract_page_no(dl_meta)
        page = self._to_page_index(page_no)

        order = self._extract_chunk_order(dl_meta)

        new_meta: Dict[str, Any] = {"source": source, "_order": order}
        if page != -1:
            new_meta["page"] = page
            new_meta["page_label"] = str(page)

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

    @staticmethod
    def _extract_chunk_order(dl_meta: Dict[str, Any]) -> int:
        items = dl_meta.get("doc_items") or []
        best: Optional[int] = None
        for it in items:
            for prov in (it.get("prov") or []):
                cs = prov.get("charspan")
                if not cs or not isinstance(cs, list) or len(cs) < 1:
                    continue
                try:
                    start = int(cs[0])
                except Exception:
                    continue
                if best is None or start < best:
                    best = start
        return best if best is not None else 0

    @staticmethod
    def _sort_key(d: Document) -> Tuple[str, int, int]:
        md = d.metadata or {}
        src = str(md.get("source") or "")
        page = int(md.get("page", -1))
        order = int(md.get("_order", 0))
        return (src, page, order)
