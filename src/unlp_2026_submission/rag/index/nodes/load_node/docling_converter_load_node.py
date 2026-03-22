from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from unlp_2026_submission.rag.index.index_state import IndexState


class DoclingConverterLoadNode:
    def __init__(
        self,
        output_suffix: str = ".md",
        zero_based_pages: bool = False,
        page_end_template: str = "\n\n<!-- page_end:{page} -->\n\n",
        join_sep: str = "\n\n",
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
        write_docling_json: bool = False,
        json_suffix: str = ".docling.json",
    ):
        self.output_suffix = output_suffix
        self.zero_based_pages = zero_based_pages
        self.page_end_template = page_end_template
        self.join_sep = join_sep

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

        self.write_docling_json = write_docling_json
        self.json_suffix = json_suffix

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
            return {"documents": []}

        out_docs: List[Document] = []

        for fp in filepaths:
            pdf_path = Path(fp)
            md_path = pdf_path.with_suffix(self.output_suffix)

            if self.selective_ocr:
                text_md = self._export_markdown_best_effort(fp, converter=self._converter_text)
                text_pages = self._split_markdown_pages(text_md)
                bad_pages = self._detect_bad_pages_from_page_map(text_pages)

                if bad_pages:
                    ocr_md = self._export_markdown_best_effort(fp, converter=self._converter_ocr)
                    ocr_pages = self._split_markdown_pages(ocr_md)
                    merged_pages = self._merge_page_maps(text_pages, ocr_pages, bad_pages)
                else:
                    merged_pages = text_pages

                md = self._build_markdown_from_pages(merged_pages)
                md_path.write_text(md, encoding="utf-8")

                out_docs.append(
                    Document(
                        page_content=md,
                        metadata={"source": str(pdf_path), "md_path": str(md_path)},
                    )
                )
            else:
                md = self._export_markdown_best_effort(fp, converter=self._converter_ocr)
                md_path.write_text(md, encoding="utf-8")
                out_docs.append(
                    Document(
                        page_content=md,
                        metadata={"source": str(pdf_path), "md_path": str(md_path)},
                    )
                )

            if self.write_docling_json:
                json_path = pdf_path.with_suffix(self.json_suffix)
                result = self._converter_ocr.convert(str(pdf_path))
                json_path.write_text(self._safe_doc_to_json(result.document), encoding="utf-8")

        return {"documents": out_docs}

    def _export_markdown_best_effort(self, filepath: str, converter: DocumentConverter) -> str:
        placeholder = "<<<DOCLING_PAGE_BREAK>>>"
        try:
            loader = DoclingLoader(
                file_path=[filepath],
                export_type=ExportType.MARKDOWN,
                converter=converter,
                md_export_kwargs={"page_break_placeholder": placeholder},
            )
            docs = loader.load()
            raw = docs[0].page_content if docs else ""
            return self._inject_page_end_from_placeholder(raw, placeholder)
        except Exception:
            chunks = self._load_doc_chunks(filepath, converter=converter)
            norm = [self._normalize_chunk(d) for d in chunks]
            norm.sort(key=self._sort_key)
            return self._chunks_to_markdown(norm)

    def _inject_page_end_from_placeholder(self, md: str, placeholder: str) -> str:
        parts = md.split(placeholder)
        out: List[str] = []
        start = 0 if self.zero_based_pages else 1
        for i, part in enumerate(parts):
            out.append(part.rstrip())
            page = start + i
            out.append(self.page_end_template.format(page=page))
        return "\n".join(out).strip()

    def _split_markdown_pages(self, markdown_text: str) -> Dict[int, str]:
        marker_re = re.compile(r"<!--\s*page_end:(-?\d+)\s*-->")
        pages: Dict[int, str] = {}

        pos = 0
        last_page: Optional[int] = None

        for m in marker_re.finditer(markdown_text):
            page_text = markdown_text[pos:m.start()].strip()
            page_no = int(m.group(1))
            pages[page_no] = page_text
            pos = m.end()
            last_page = page_no

        tail = markdown_text[pos:].strip()
        if tail:
            fallback_page = (last_page + 1) if last_page is not None else (0 if self.zero_based_pages else 1)
            pages[fallback_page] = tail

        return pages

    def _build_markdown_from_pages(self, pages: Dict[int, str]) -> str:
        out: List[str] = []
        for page_no in sorted(pages.keys()):
            page_text = (pages[page_no] or "").strip()
            if not page_text:
                continue
            out.append(page_text)
            out.append(self.page_end_template.format(page=page_no))
        return "\n".join(out).strip()

    def _merge_page_maps(
        self,
        text_pages: Dict[int, str],
        ocr_pages: Dict[int, str],
        bad_pages: set[int],
    ) -> Dict[int, str]:
        merged: Dict[int, str] = {}
        all_pages = sorted(set(text_pages.keys()) | set(ocr_pages.keys()))

        for page_no in all_pages:
            if page_no in bad_pages and page_no in ocr_pages:
                merged[page_no] = ocr_pages[page_no]
            else:
                merged[page_no] = text_pages.get(page_no, ocr_pages.get(page_no, ""))

        return merged

    def _detect_bad_pages_from_page_map(self, pages: Dict[int, str]) -> set[int]:
        bad: set[int] = set()

        for page_no, text in pages.items():
            txt = (text or "").strip()
            if not txt:
                continue

            n = len(txt)
            alnum = sum(1 for ch in txt if ch.isalnum())
            ratio = (alnum / n) if n else 0.0

            if n < self.selective_min_page_chars or ratio < self.selective_min_alnum_ratio:
                bad.add(page_no)

        return bad

    def _chunks_to_markdown(self, chunks: List[Document]) -> str:
        pages: DefaultDict[Tuple[str, int], List[Document]] = defaultdict(list)
        for d in chunks:
            md = d.metadata or {}
            src = str(md.get("source") or "")
            page = int(md.get("page", -1))
            pages[(src, page)].append(d)

        out: List[str] = []
        for (src, page), items in sorted(pages.items(), key=lambda x: (x[0][0], x[0][1])):
            items.sort(key=self._sort_key)
            page_text = self._build_page_text_without_heading_repeats(items)
            if not page_text:
                continue
            out.append(page_text)
            out.append(self.page_end_template.format(page=page))

        return "\n".join(out).strip()

    def _build_page_text_without_heading_repeats(self, items: List[Document]) -> str:
        text_parts: List[str] = []
        page_norm_text = ""
        seen_chunks: set[str] = set()
        current_heading: Optional[str] = None

        for it in items:
            chunk = (it.page_content or "").strip()
            if not chunk:
                continue

            chunk, current_heading = self._drop_repeated_heading_prefix(chunk, current_heading)
            if not chunk:
                continue

            if text_parts:
                chunk = self._trim_overlap_with_previous_text(text_parts[-1], chunk)
                if not chunk:
                    continue

            chunk_norm = self._normalize_for_dedup(chunk)
            if not chunk_norm:
                continue

            # Skip exact and large in-page duplicates caused by doc chunk overlap.
            if chunk_norm in seen_chunks:
                continue
            if len(chunk_norm) >= 80 and chunk_norm in page_norm_text:
                continue

            text_parts.append(chunk)
            seen_chunks.add(chunk_norm)
            page_norm_text = f"{page_norm_text} {chunk_norm}".strip()

        return self.join_sep.join(text_parts).strip()

    def _drop_repeated_heading_prefix(
        self,
        text: str,
        current_heading: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        lines = text.splitlines()
        first_idx: Optional[int] = None
        for i, ln in enumerate(lines):
            if ln.strip():
                first_idx = i
                break

        if first_idx is None:
            return "", current_heading

        first_line = lines[first_idx].strip()
        has_tail = any(ln.strip() for ln in lines[first_idx + 1 :])

        # Docling chunks can repeat the same section heading in each chunk.
        if current_heading is not None and has_tail and first_line == current_heading:
            lines[first_idx] = ""
            while lines and not lines[0].strip():
                lines.pop(0)
            return "\n".join(lines).strip(), current_heading

        if self._is_heading_line(first_line):
            current_heading = first_line

        return text, current_heading

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        raw = line.strip()
        if not raw:
            return False
        if raw.startswith("#"):
            return True

        # Numeric and roman section-style headings, including Ukrainian "І.".
        if re.match(r"^\d+(?:\.\d+)*[.)]\s+\S", raw):
            return True
        if re.match(r"^[IVXLCDMІVXLCDM]+[.)]\s+\S", raw, flags=re.IGNORECASE):
            return True
        if DoclingConverterLoadNode._is_uppercase_heading_like(raw):
            return True

        return False

    @staticmethod
    def _is_uppercase_heading_like(raw: str) -> bool:
        letters = [ch for ch in raw if ch.isalpha()]
        if len(letters) < 8:
            return False

        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        # High-level headings from OCR/chunks are often fully upper-cased.
        return upper_ratio >= 0.85 and len(raw) <= 220

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        return " ".join(text.lower().split())

    def _trim_overlap_with_previous_text(self, prev_text: str, curr_text: str) -> str:
        prev_tokens = re.findall(r"\S+", prev_text.lower())
        curr_tokens = re.findall(r"\S+", curr_text.lower())
        if not prev_tokens or not curr_tokens:
            return curr_text.strip()

        max_overlap = min(len(prev_tokens), len(curr_tokens), 200)
        min_overlap = 12

        overlap = 0
        for k in range(max_overlap, min_overlap - 1, -1):
            if prev_tokens[-k:] == curr_tokens[:k]:
                overlap = k
                break

        if overlap == 0:
            return curr_text.strip()

        return self._drop_leading_tokens(curr_text, overlap)

    @staticmethod
    def _drop_leading_tokens(text: str, tokens_to_drop: int) -> str:
        if tokens_to_drop <= 0:
            return text.strip()

        for i, match in enumerate(re.finditer(r"\S+", text)):
            if i == tokens_to_drop:
                return text[match.start() :].lstrip()

        return ""

    def _load_doc_chunks(self, filepath: str, converter: DocumentConverter) -> List[Document]:
        loader = DoclingLoader(
            file_path=[filepath],
            export_type=ExportType.DOC_CHUNKS,
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
                from docling.datamodel.pipeline_options import smolvlm_picture_description

                _setattr_if(pipeline_options, "picture_description_options", smolvlm_picture_description)
        else:
            _setattr_if(pipeline_options, "do_picture_description", False)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def _normalize_chunk(self, doc: Document) -> Document:
        md = doc.metadata or {}
        source = md.get("source") or md.get("file_path") or md.get("path") or "unknown"

        dl_meta = md.get("dl_meta") or {}
        page_no = self._extract_page_no(dl_meta)
        page = self._to_page_index(page_no)
        order = self._extract_chunk_order(dl_meta)

        new_meta: Dict[str, Any] = {"source": source, "page": page, "_order": order}
        if page != -1:
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

    @staticmethod
    def _safe_doc_to_json(doc: Any) -> str:
        import json

        try:
            return json.dumps(doc.export_to_dict(), ensure_ascii=False)
        except Exception:
            try:
                return json.dumps(doc.model_dump(), ensure_ascii=False)
            except Exception:
                return "{}"
