from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
from pathlib import Path

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
    predicted_document_id = workflow_result['relevant_document_id']
    correct_document_id = question['doc_id']
    predicted_document_id_stem = Path(predicted_document_id).stem
    correct_document_id_stem = Path(correct_document_id).stem

    d_i = 1.0 if predicted_document_id_stem == correct_document_id_stem else 0.0

    # --- p_i: page proximity ---
    p_i = 0.0
    if predicted_document_id_stem == correct_document_id_stem:
        predicted_document_page = workflow_result['relevant_document_page_num']
        correct_document_page = question['page_num']
        n_pages = question['n_pages']

        if predicted_document_page is None or correct_document_page is None or not n_pages or n_pages <= 0:
            p_i = 0.0
        else:
            p_i = 1.0 - (abs(predicted_document_page - correct_document_page) / float(n_pages))
            p_i = max(0.0, min(1.0, p_i))

    score = 0.5 * d_i + 0.5 * p_i

    return MetricResult(max(0.0, min(1.0, score)))

def calculate_total_documents_source_composite_accuracy(
        experiment_results: Experiment
):
    total = len(experiment_results)
    total_score = sum(r["score"] for r in experiment_results)
    accuracy = total_score / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'total_score': total_score,
    }
