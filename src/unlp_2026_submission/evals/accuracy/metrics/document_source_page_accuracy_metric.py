from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
from pathlib import Path

from unlp_2026_submission.entities import Question
from unlp_2026_submission.workflow.state import WorkflowState

from .accuracy_metric_name import AccuracyMetricName


@numeric_metric(
    name=AccuracyMetricName.DOCUMENT_SOURCE_PAGES.value,
    allowed_values=(0.0, 1.0)
)
def document_source_page_accuracy_metric(
        question: Question,
        workflow_result: WorkflowState,
) -> MetricResult:
    """Calculate accuracy of the prediction."""
    predicted_document_id_stem = Path(workflow_result['reference_document_id']).stem
    correct_document_id_stem = Path(question['doc_id']).stem

    if correct_document_id_stem != predicted_document_id_stem:
        return MetricResult(0, 'not correct document_id')

    reference_document_page_num = workflow_result['reference_document_page_num']
    actual_document_page = question['page_num']
    n_pages = question['n_pages']

    try:
        reference_document_page_num = int(reference_document_page_num)
        actual_document_page = int(actual_document_page)
        n_pages = int(n_pages)
    except (TypeError, ValueError):
        return MetricResult(value=0.0, reason="invalid page data")

    if n_pages <= 0:
        return MetricResult(value=0.0, reason="invalid n_pages")

    metric_value = 1.0 - abs(reference_document_page_num - actual_document_page) / float(n_pages)
    metric_value = max(0.0, min(1.0, float(metric_value)))

    return MetricResult(value=metric_value)

def calculate_total_documents_source_page_accuracy(
        experiment_results: Experiment
):
    total = len(experiment_results)
    total_score = sum((r["score"] or 0) for r in experiment_results)
    accuracy = total_score / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'total_score': total_score,
    }
