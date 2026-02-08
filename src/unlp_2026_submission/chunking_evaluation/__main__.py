"""
/unlp-2026-submission/src/unlp_2026_submission/chunking_evaluation/__main__.py

Page-aware semantic chunking pipeline (stable) using LlamaIndex SimpleDirectoryReader:

1) Load .txt files via SimpleDirectoryReader (LlamaIndex parsing)
2) For each file, parse "===== Page N =====" delimiters that appear at END of each page:
   - build clean_text (delimiters removed, pages joined by "\\n\\n")
   - build page_ranges = [(start_char, end_char, page_number), ...] in clean_text
3) Run semantic chunking via existing ClusterSemanticChunker on clean_text
4) For each semantic chunk, find its span stably (whitespace-insensitive + fuzzy sentence fallback)
   and assign page_number = page that overlaps the chunk span the most.
5) Return chunk Documents with metadata:
   {"page_label": page_num, "file_name": <name>, "start": <int|None>, "end": <int|None>}
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from chromadb.utils import embedding_functions
from llama_index.core import Document, SimpleDirectoryReader

from unlp_2026_submission.config import Config
from unlp_2026_submission.chunking_evaluation.chunking import ClusterSemanticChunker


# Delimiter example at end of each page:
#   "===== Page 12 ====="
PAGE_DELIMITER_RE = re.compile(r"===== Page (\d+) =====")


# -----------------------------
# Step 1: Parse raw text into clean_text + page_ranges
# -----------------------------
def load_text_with_pages(raw: str) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Reads raw text where the delimiter "===== Page N =====" appears at the END of page N.

    Returns:
      - clean_text: full text with delimiters removed (pages joined by "\\n\\n")
      - page_ranges: list of (start_char, end_char, page_number) spans in clean_text
    """
    matches = list(PAGE_DELIMITER_RE.finditer(raw))
    if not matches:
        clean = raw.strip()
        return clean, [(0, len(clean), 1)]

    pages: List[Tuple[int, str]] = []
    last_end = 0
    last_page_num: Optional[int] = None

    for m in matches:
        page_num = int(m.group(1))
        page_text = raw[last_end : m.start()].strip()
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
def dominant_page_for_span(
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


# -----------------------------
# Stable chunk span search
# -----------------------------
def _normalize_simple(text: str) -> str:
    """Whitespace-collapse + strip (no mapping needed)."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_with_map(text: str) -> tuple[str, List[int]]:
    """
    Normalize whitespace to single spaces, and return:
      - normalized_text
      - norm2orig: mapping from each normalized char index -> original char index
    """
    norm_chars: List[str] = []
    norm2orig: List[int] = []

    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                norm_chars.append(" ")
                norm2orig.append(i)  # map inserted space to first ws char
                in_ws = True
        else:
            norm_chars.append(ch)
            norm2orig.append(i)
            in_ws = False

    # Trim leading/trailing spaces in normalized representation (keep mapping consistent)
    while norm_chars and norm_chars[0] == " ":
        norm_chars.pop(0)
        norm2orig.pop(0)
    while norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        norm2orig.pop()

    return "".join(norm_chars), norm2orig


def _token_sort_ratio(a: str, b: str) -> float:
    """
    Lightweight replacement for fuzz.token_sort_ratio using difflib.
    Returns 0..100.
    """
    ta = " ".join(sorted(_normalize_simple(a).lower().split()))
    tb = " ".join(sorted(_normalize_simple(b).lower().split()))
    return 100.0 * SequenceMatcher(None, ta, tb).ratio()


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Split into sentence-like spans but keep exact (start,end) in original string.
    Splits on punctuation + whitespace OR newlines.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"([.!?]+)\s+|\n+", text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def rigorous_document_search_span(
    document: str,
    target: str,
    min_score: float = 98.0,
) -> Optional[Tuple[int, int]]:
    """
    Stable search for target inside document.
    1) Exact search (fast)
    2) Whitespace-insensitive search using normalized doc + index map (stable)
    3) Sentence-span fuzzy fallback (stable indices), thresholded

    Returns (start,end) in ORIGINAL document, or None.
    """
    t = target.rstrip(".")  # small tolerance

    # 1) Exact
    pos = document.find(t)
    if pos != -1:
        return pos, pos + len(t)

    # 2) Whitespace-insensitive with mapping
    doc_norm, norm2orig = _normalize_with_map(document)
    t_norm = _normalize_simple(t)

    if t_norm:
        posn = doc_norm.find(t_norm)
        if posn != -1:
            start = norm2orig[posn]
            last_norm_idx = min(posn + len(t_norm) - 1, len(norm2orig) - 1)
            end = norm2orig[last_norm_idx] + 1
            return start, end

    # 3) Fuzzy sentence fallback
    spans = _sentence_spans(document)
    best_span: Optional[Tuple[int, int]] = None
    best_score = -1.0

    for (s, e) in spans:
        sent = document[s:e].strip()
        if not sent:
            continue
        score = _token_sort_ratio(t, sent)
        if score > best_score:
            best_score = score
            best_span = (s, e)

    if best_span is None or best_score < min_score:
        return None

    return best_span


# -----------------------------
# Attach page metadata to chunks (return Documents)
# -----------------------------
def add_page_metadata_to_chunks_from_filename(
    clean_text: str,
    chunks: List[str],
    page_ranges: List[Tuple[int, int, int]],
    file_name: str,
    min_fuzzy_score: float = 98.0,
) -> List[Document]:
    """
    For each chunk, locate it in clean_text (stable),
    assign dominant page_number, and return LlamaIndex Documents.

    NOTE: start/end are stored in metadata (Document has no .start/.end attrs).
    """
    documents: List[Document] = []
    hint = 0

    for chunk in chunks:
        suffix = clean_text[hint:]
        span = rigorous_document_search_span(suffix, chunk, min_score=min_fuzzy_score)

        if span is not None:
            start, end = span[0] + hint, span[1] + hint
        else:
            span2 = rigorous_document_search_span(clean_text, chunk, min_score=min_fuzzy_score)
            if span2 is None:
                documents.append(
                    Document(
                        text=chunk,
                        metadata={
                            "page_label": None,
                            "file_name": file_name,
                            "start": None,
                            "end": None,
                        },
                    )
                )
                continue
            start, end = span2

        page_num = dominant_page_for_span(start, end, page_ranges)

        documents.append(
            Document(
                text=chunk,
                metadata={
                    "page_label": page_num,
                    "file_name": file_name,
                    "start": start,
                    "end": end,
                },
            )
        )

        hint = max(hint, start + 1)

    return documents


def _infer_file_name_from_doc(doc: Document) -> str:
    md = doc.metadata or {}
    return (
        md.get("file_name")
        or md.get("filename")
        or (Path(md["file_path"]).name if "file_path" in md else None)
        or "unknown.txt"
    )


def run_page_aware_semantic_chunking_from_document(
    source_doc: Document,
    chunker: ClusterSemanticChunker,
    min_fuzzy_score: float = 98.0,
) -> List[Document]:
    raw_text = source_doc.text or ""
    file_name = _infer_file_name_from_doc(source_doc)

    clean_text, page_ranges = load_text_with_pages(raw_text)
    chunks = chunker.split_text(clean_text)

    return add_page_metadata_to_chunks_from_filename(
        clean_text=clean_text,
        chunks=chunks,
        page_ranges=page_ranges,
        file_name=file_name,
        min_fuzzy_score=min_fuzzy_score,
    )


# -----------------------------
# CLI entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Page-aware semantic chunking (stable span mapping) via SimpleDirectoryReader")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit list of .txt files. If omitted, falls back to a default test file under Config.data_root_dir.",
    )
    parser.add_argument("--max-chunk", type=int, default=1600)
    parser.add_argument("--min-chunk", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--preview", type=int, default=10, help="How many chunks to print")
    parser.add_argument("--min-fuzzy-score", type=float, default=98.0, help="Threshold for fuzzy fallback (0..100)")
    args = parser.parse_args()

    config = Config()
    data_dir = Path(config.data_root_dir)

    # Resolve input files
    if args.files and len(args.files) > 0:
        input_files = [str(Path(p)) for p in args.files]
    else:
        default_file = data_dir / "test" / "0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.txt"
        if not default_file.exists():
            default_file = data_dir / "0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.txt"
        if not default_file.exists():
            raise FileNotFoundError(
                f"Default input file not found: {default_file}\n"
                f"Provide --files /path/a.txt /path/b.txt"
            )
        input_files = [str(default_file)]

    # Load via LlamaIndex reader
    reader = SimpleDirectoryReader(
        input_dir=config.data_root_dir,
        recursive=True,
        # required_exts=['.txt'],
        input_files=[
            '/Users/kolanosenko/Projects/unlp-2026-submission/src/data/test/t1.txt',
            '/Users/kolanosenko/Projects/unlp-2026-submission/src/data/test/t2.txt',
        ]
    )
    source_documents = reader.load_data()

    # Embedding function for your ClusterSemanticChunker
    embedding_function = embedding_functions.GoogleGenaiEmbeddingFunction(
        model_name="gemini-embedding-001"
    )

    chunker = ClusterSemanticChunker(
        max_chunk_size=args.max_chunk,
        min_chunk_size=args.min_chunk,
        embedding_function=embedding_function,
        batch_size=args.batch_size,
    )

    # Run pipeline per loaded doc
    all_chunk_docs: List[Document] = []
    for src in source_documents:
        all_chunk_docs.extend(
            run_page_aware_semantic_chunking_from_document(
                src,
                chunker,
                min_fuzzy_score=args.min_fuzzy_score,
            )
        )

    print("Loaded files:", len(source_documents))
    for d in source_documents:
        print(" -", _infer_file_name_from_doc(d))
    print("Total chunks:", len(all_chunk_docs))

    # Preview
    n = min(args.preview, len(all_chunk_docs))
    for i in range(n):
        d = all_chunk_docs[i]
        meta = d.metadata or {}
        print(
            f"\n--- Chunk {i+1} | "
            f"file={meta.get('file_name')} | "
            f"page_number={meta.get('page_label')} | "
            f"span=({meta.get('start')},{meta.get('end')}) ---"
        )
        print(d.text)

    if len(all_chunk_docs) > n:
        print(f"\n... and {len(all_chunk_docs) - n} more chunks")


if __name__ == "__main__":
    main()
