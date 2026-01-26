from pathlib import Path
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import re
from rapidfuzz import fuzz

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _prepare_dataframe(
        df: pd.DataFrame,
        similarity_threshold: int = 97
):
    """
    Deduplicate page texts using token-based similarity.

    Returns:
      - unique_pages_df: DataFrame containing only unique (representative) pages
      - row_index_to_representative_index: mapping from original df.index
        to the df.index of the representative page
    """
    # Original page texts as strings
    page_texts = df["Page_Text"].astype(str).tolist()

    # Normalized texts for similarity comparison
    normalized_page_texts = [_normalize_text(text) for text in page_texts]

    # Positions (0..n-1) of representative / unique pages
    representative_positions: list[int] = []

    # Maps each row position to its representative row position
    row_pos_to_representative_pos: dict[int, int] = {}

    # Iterate over all pages
    for current_pos, current_text in enumerate(normalized_page_texts):
        best_matching_representative_pos = None
        highest_similarity_score = -1

        # Compare against already selected representatives
        for representative_pos in representative_positions:
            similarity_score = fuzz.token_set_ratio(
                current_text,
                normalized_page_texts[representative_pos],
            )

            if similarity_score > highest_similarity_score:
                highest_similarity_score = similarity_score
                best_matching_representative_pos = representative_pos

        # Decide whether current page is a duplicate
        if highest_similarity_score >= similarity_threshold:
            # Map current page to existing representative
            row_pos_to_representative_pos[current_pos] = (
                best_matching_representative_pos
            )
        else:
            # Treat current page as a new unique representative
            representative_positions.append(current_pos)
            row_pos_to_representative_pos[current_pos] = current_pos

    # Convert position-based mapping to DataFrame index-based mapping
    row_index_to_representative_index = {
        df.index[row_position]:
        df.index[row_pos_to_representative_pos[row_position]]
        for row_position in range(len(df))
    }

    # Build DataFrame of unique pages
    unique_pages_df = df.iloc[representative_positions].copy()

    return unique_pages_df, row_index_to_representative_index

def _build_retrieval_hf_dataset() -> dict[str, Dataset]:
    datasets_root_dir = Path(__file__).resolve().parent

    original_df = pd.read_csv(os.path.join(datasets_root_dir, "qa_questions.csv"))

    # --- Dedup pages ---
    unique_pages_df, row_to_representative_index_map = _prepare_dataframe(original_df)

    # --- Build corpus ---
    # use representative df index for stable doc ids
    corpus_rows = []
    representative_row_to_docid = {}
    for row in unique_pages_df.itertuples(index=True):
        representative_row_index = row.Index
        corpus_document_id = f"C{representative_row_index}"

        representative_row_to_docid[representative_row_index] = corpus_document_id

        corpus_rows.append({
            "id": corpus_document_id,
            "title": "",
            "text": str(getattr(row, "Page_Text")),
        })

    corpus_ds = Dataset.from_list(corpus_rows)

    # --- Build queries ---
    queries_rows = []
    for row in original_df.itertuples(index=False):
        qid = f"Q{getattr(row, 'Question_ID')}"

        queries_rows.append({
            "id": qid,
            "text": str(getattr(row, "Question"))
        })

    queries_ds = Dataset.from_list(queries_rows)

    # --- Build qrels ---
    qrels_rows = []
    # If you can have multiple correct pages per question, add them all here.
    for row in original_df.itertuples(index=True):
        qid = f"Q{getattr(row, 'Question_ID')}"

        rep_idx = row_to_representative_index_map[row.Index]
        doc_id = representative_row_to_docid[rep_idx]
        qrels_rows.append({"query-id": qid, "corpus-id": doc_id, "score": 1})

    qrels_ds = Dataset.from_list(qrels_rows)

    return {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels": qrels_ds,
    }

def get_qa_mteb_dataset() -> dict[str, Dataset]:
    dataset = _build_retrieval_hf_dataset()

    return dataset
