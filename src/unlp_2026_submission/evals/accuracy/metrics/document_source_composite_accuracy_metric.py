from pathlib import Path

from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

from unlp_2026_submission.entities import Question
from unlp_2026_submission.workflow.state import QAWorkflowState

from .accuracy_metric_name import AccuracyMetricName


@numeric_metric(
    name=AccuracyMetricName.DOCUMENT_SOURCES_COMPOSITE.value,
    allowed_values=(0.0, 1.0)
)
def document_source_composite_accuracy_metric(
    question: Question,
    workflow_result: QAWorkflowState,
) -> MetricResult:
    """
        composite score = 0.5*d_i + 0.5*p_i
    """
    # --- d_i: correct document id ---
    predicted_document_id = workflow_result["relevant_document_id"]
    correct_document_id = question["doc_id"]
    predicted_document_id_stem = Path(predicted_document_id).stem
    correct_document_id_stem = Path(correct_document_id).stem

    d_i = 1.0 if predicted_document_id_stem == correct_document_id_stem else 0.0

    # --- p_i: page proximity ---
    p_i = 0.0
    if predicted_document_id_stem == correct_document_id_stem:
        predicted_document_page = workflow_result["relevant_document_page_num"]
        correct_document_page = question["page_num"]
        n_pages = question["n_pages"]

        try:
            predicted_document_page = int(predicted_document_page)
            correct_document_page = int(correct_document_page)
            n_pages = int(n_pages)
        except (TypeError, ValueError):
            p_i = 0.0
        else:
            if n_pages <= 0:
                p_i = 0.0
            else:
                p_i = 1.0 - (abs(predicted_document_page - correct_document_page) / float(n_pages))
                p_i = max(0.0, min(1.0, float(p_i)))

    score = 0.5 * float(d_i) + 0.5 * float(p_i)
    score = max(0.0, min(1.0, score))

    return MetricResult(value=score)


def _document_id_stem(doc_id: str) -> str:
    return Path(doc_id).stem if doc_id else ""


def _document_source_correct_for_row(r: dict) -> int:
    """1 if correct document id, 0 otherwise (for composite experiment_view rows)."""
    correct_stem = _document_id_stem(r.get("correct_document_id") or "")
    pred_stem = _document_id_stem(r.get("relevant_document_id") or "")
    return 1 if correct_stem == pred_stem else 0


def _page_score_for_row(r: dict) -> float:
    """Page accuracy score for one row (same logic as document_source_page_accuracy_metric)."""
    correct_stem = _document_id_stem(r.get("correct_document_id") or "")
    pred_stem = _document_id_stem(r.get("relevant_document_id") or "")
    if correct_stem != pred_stem:
        return 0.0
    n_pages = r.get("n_pages") or 1
    pred_page = r.get("relevant_document_page_num", 0)
    actual_page = r.get("correct_document_page", 0)
    return 1.0 - abs(pred_page - actual_page) / n_pages


def _recall_at_k(experiment_results: Experiment, k: int) -> float:
    """Percentage of rows where (correct_document_id, correct_document_page) is in top_k of relevant_documents."""
    total = len(experiment_results)
    if total == 0:
        return 0.0
    hits = 0
    for r in experiment_results:
        correct_stem = _document_id_stem(r.get("correct_document_id") or "")
        correct_page = r.get("correct_document_page")
        if correct_page is None:
            continue
        try:
            correct_page = int(correct_page)
        except (TypeError, ValueError):
            continue
        relevant = r.get("relevant_documents") or []
        top_k = relevant[:k]
        for item in top_k:
            doc_id = item.get("document_id")
            page = item.get("page")
            if doc_id is None or page is None:
                continue
            try:
                page = int(page)
            except (TypeError, ValueError):
                continue
            if _document_id_stem(doc_id) == correct_stem and page == correct_page:
                hits += 1
                break
    return hits / total


def _metrics_by_domain(experiment_results: Experiment):
    """Compute document_source and document_page metrics grouped by question_domain."""
    by_domain: dict[str, list[dict]] = {}
    for r in experiment_results:
        domain = r.get("question_domain")
        if domain is None or (isinstance(domain, str) and domain.strip() == ""):
            domain = "unknown"
        else:
            domain = str(domain).strip()
        by_domain.setdefault(domain, []).append(r)

    doc_source_by_domain = {}
    doc_page_by_domain = {}
    for domain, rows in by_domain.items():
        n = len(rows)
        doc_correct = sum(_document_source_correct_for_row(row) for row in rows)
        doc_source_by_domain[domain] = {
            "accuracy": doc_correct / n if n > 0 else 0.0,
            "correct": doc_correct,
            "total": n,
        }
        page_total_score = sum(_page_score_for_row(row) for row in rows)
        doc_page_by_domain[domain] = {
            "accuracy": page_total_score / n if n > 0 else 0.0,
            "total_score": page_total_score,
            "total": n,
        }
    return doc_source_by_domain, doc_page_by_domain


def calculate_total_documents_source_composite_accuracy(
    experiment_results: Experiment,
):
    total = len(experiment_results)
    composite_total_score = sum(r["score"] for r in experiment_results)
    composite_accuracy = composite_total_score / total if total > 0 else 0.0

    doc_correct = sum(_document_source_correct_for_row(r) for r in experiment_results)
    doc_accuracy = doc_correct / total if total > 0 else 0.0

    page_total_score = sum(_page_score_for_row(r) for r in experiment_results)
    page_accuracy = page_total_score / total if total > 0 else 0.0

    doc_source_by_domain, doc_page_by_domain = _metrics_by_domain(experiment_results)

    return {
        "accuracy": composite_accuracy,
        "total_score": composite_total_score,
        "document_source_accuracy": {
            "total": {"accuracy": doc_accuracy, "correct": doc_correct, "total": total},
            "by_domain": doc_source_by_domain,
        },
        "document_page_source_accuracy": {
            "total": {
                "accuracy": page_accuracy,
                "total_score": page_total_score,
                "total": total,
            },
            "by_domain": doc_page_by_domain,
        },
        "recall_at_1": _recall_at_k(experiment_results, 1),
        "recall_at_3": _recall_at_k(experiment_results, 3),
        "recall_at_5": _recall_at_k(experiment_results, 5),
        "recall_at_10": _recall_at_k(experiment_results, 10),
    }
