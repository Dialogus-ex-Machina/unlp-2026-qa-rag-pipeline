from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

from unlp_2026_submission.entities import SingleAnswerQuestion
from unlp_2026_submission.workflow import WorkflowState


@numeric_metric(name="document_source_page_accuracy", allowed_values=(0.0, 1.0))
def document_source_page_accuracy_metric(
        question: SingleAnswerQuestion,
        workflow_result: WorkflowState,
) -> MetricResult:
    """Calculate accuracy of the prediction."""
    if question['doc_id'] != workflow_result['reference_document_id']:
        return MetricResult(0, 'not correct document_id')

    reference_document_page = workflow_result['reference_document_page']
    actual_document_page = question['page_num']
    n_pages = question['n_pages']

    metric_value = 1 - abs(reference_document_page - actual_document_page) / n_pages

    return MetricResult(metric_value)

def calculate_total_documents_source_page_accuracy(
        experiment_results: Experiment
):
    total = len(experiment_results)
    total_score = sum(r["score"] for r in experiment_results)
    accuracy = total_score / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'total_score': total_score,
    }
