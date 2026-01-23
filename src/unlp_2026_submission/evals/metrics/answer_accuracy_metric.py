from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

from unlp_2026_submission.evals.evaluation_metric_name import EvaluationMetricName


@numeric_metric(
    name=EvaluationMetricName.ANSWER_ACCURACY.value,
    allowed_values=(0.0, 1.0)
)
def answer_accuracy_metric(prediction: str, actual: str) -> MetricResult:
    """Calculate accuracy of the prediction."""
    return (
        MetricResult(value=1, reason="")
        if prediction == actual
        else MetricResult(value=0, reason="")
    )


def calculate_total_answers_accuracy(experiment_results: Experiment):
    total = len(experiment_results)
    correct = sum(1 for r in experiment_results if r["score"] == 1)
    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
    }
