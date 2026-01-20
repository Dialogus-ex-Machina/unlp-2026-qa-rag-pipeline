from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

from unlp_2026_submission.entities import SingleAnswerQuestion
from unlp_2026_submission.workflow import WorkflowState


@numeric_metric(name="composite_accuracy", allowed_values=(0.0, 1.0))
def composite_accuracy_metric(
    question: SingleAnswerQuestion,
    workflow_result: WorkflowState,
) -> MetricResult:
    """
        composite score = 0.5*a_i + 0.25*d_i + 0.25*p_i
    """

    # --- a_i: correct answer ---
    predicted_answer = workflow_result['answer']
    correct_answer = question['correct_answer']

    a_i = 1.0 if predicted_answer == correct_answer else 0.0

    # --- d_i: correct document id ---
    predicted_document_id = workflow_result['reference_document_id']
    correct_document_id = question['doc_id']

    d_i = 1.0 if correct_document_id == predicted_document_id else 0.0

    # --- p_i: page proximity ---
    p_i = 0.0
    if correct_document_id == predicted_document_id:
        predicted_document_page = workflow_result['reference_document_page']
        correct_document_page = question['page_num']
        n_pages = question['n_pages']

        if predicted_document_page is None or correct_document_page is None or not n_pages or n_pages <= 0:
            p_i = 0.0
        else:
            p_i = 1.0 - (abs(predicted_document_page - correct_document_page) / float(n_pages))
            p_i = max(0.0, min(1.0, p_i))

    score = 0.5 * a_i + 0.25 * d_i + 0.25 * p_i

    return MetricResult(max(0.0, min(1.0, score)))

def calculate_total_composite_accuracy(
        experiment_results: Experiment
):
    total = len(experiment_results)
    total_score = sum(r["score"] for r in experiment_results)
    accuracy = total_score / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'total_score': total_score,
    }
