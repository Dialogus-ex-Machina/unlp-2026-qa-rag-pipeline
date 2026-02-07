"""
/unlp-2026-submission/src/unlp_2026_submission/chunking_evaluation/__main__.py

Page-aware semantic chunking pipeline (stable):

1) Load a .txt where page delimiter appears at END of each page:
   "===== Page N ====="
2) Remove delimiters, build:
   - clean_text
   - page_ranges = [(start_char, end_char, page_number), ...] in clean_text
3) Run semantic chunking via existing ClusterSemanticChunker
4) For each semantic chunk, find its span stably (whitespace-insensitive + fuzzy sentence fallback)
   and assign page_number = page that overlaps the chunk span the most.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from chromadb.utils import embedding_functions

from unlp_2026_submission.config import Config
from unlp_2026_submission.chunking_evaluation.chunking import ClusterSemanticChunker


# Delimiter example at end of each page:
#   "===== Page 12 ====="
PAGE_DELIMITER_RE = re.compile(r"===== Page (\d+) =====")


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class ChunkWithMetadata:
    text: str
    page_number: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None


# -----------------------------
# Step 1: Load & remove delimiters + page ranges
# -----------------------------
def load_file_with_pages(file_path: str | Path) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Reads a .txt where the delimiter "===== Page N =====" appears at the END of page N.

    Returns:
      - clean_text: full text with delimiters removed (pages joined by "\\n\\n")
      - page_ranges: list of (start_char, end_char, page_number) spans in clean_text
    """
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8")

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


def rigorous_document_search_span(document: str, target: str, min_score: float = 98.0) -> Optional[Tuple[int, int]]:
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
# Attach page metadata to chunks
# -----------------------------
def add_page_metadata_to_chunks(
    clean_text: str,
    chunks: List[str],
    page_ranges: List[Tuple[int, int, int]],
    min_fuzzy_score: float = 98.0,
) -> List[ChunkWithMetadata]:
    """
    For each chunk, locate it in clean_text (stable), then assign dominant page_number.
    """
    out: List[ChunkWithMetadata] = []
    hint = 0  # helps keep search monotonic

    for chunk in chunks:
        # Prefer searching from hint forward (performance + stability)
        suffix = clean_text[hint:]
        span = rigorous_document_search_span(suffix, chunk, min_score=min_fuzzy_score)

        if span is not None:
            start, end = span[0] + hint, span[1] + hint
        else:
            span2 = rigorous_document_search_span(clean_text, chunk, min_score=min_fuzzy_score)
            if span2 is None:
                out.append(ChunkWithMetadata(text=chunk, page_number=None))
                continue
            start, end = span2

        page_num = dominant_page_for_span(start, end, page_ranges)
        out.append(ChunkWithMetadata(text=chunk, page_number=page_num, start=start, end=end))
        hint = max(hint, start + 1)

    return out


def run_page_aware_semantic_chunking(
    file_path: str | Path,
    chunker: ClusterSemanticChunker,
    min_fuzzy_score: float = 98.0,
) -> List[ChunkWithMetadata]:
    """
    Full pipeline:
    1) Load file with page delimiters, remove delimiters, build page ranges.
    2) Semantic chunking on clean_text.
    3) Attach dominant page_number per chunk (stable span finding).
    """
    clean_text, page_ranges = load_file_with_pages(file_path)
    chunks = chunker.split_text(clean_text)
    return add_page_metadata_to_chunks(clean_text, chunks, page_ranges, min_fuzzy_score=min_fuzzy_score)


# -----------------------------
# CLI entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Page-aware semantic chunking (stable span mapping)")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to .txt with '===== Page N =====' delimiters (end-of-page).",
    )
    parser.add_argument("--max-chunk", type=int, default=1600)
    parser.add_argument("--min-chunk", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--preview", type=int, default=10, help="How many chunks to print")
    parser.add_argument("--min-fuzzy-score", type=float, default=98.0, help="Threshold for fuzzy fallback (0..100)")
    args = parser.parse_args()

    config = Config()
    data_dir = Path(config.data_root_dir)

    if args.file:
        input_file = Path(args.file)
    else:
        input_file = data_dir / "test" / "0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.txt"
        if not input_file.exists():
            input_file = data_dir / "0ec2a844ab5e5f66bfff7d66dbd5b33bca34cc94.txt"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Provide --file /path/to/file.txt or fix Config.data_root_dir."
        )

    # You can swap this out for your own embedding function if needed.
    embedding_function = embedding_functions.GoogleGenaiEmbeddingFunction(
        model_name="gemini-embedding-001"
    )

    chunker = ClusterSemanticChunker(
        max_chunk_size=args.max_chunk,
        min_chunk_size=args.min_chunk,
        embedding_function=embedding_function,
        batch_size=args.batch_size,
    )

    chunks_with_meta = run_page_aware_semantic_chunking(
        input_file,
        chunker,
        min_fuzzy_score=args.min_fuzzy_score,
    )

    print("Input:", input_file)
    print("Chunks:", len(chunks_with_meta))

    n = min(args.preview, len(chunks_with_meta))
    for i in range(n):
        c = chunks_with_meta[i]
        print(f"\n--- Chunk {i+1} | page_number={c.page_number} | span=({c.start},{c.end}) ---")
        print(c.text)

    if len(chunks_with_meta) > n:
        print(f"\n... and {len(chunks_with_meta) - n} more chunks")


if __name__ == "__main__":
    main()
