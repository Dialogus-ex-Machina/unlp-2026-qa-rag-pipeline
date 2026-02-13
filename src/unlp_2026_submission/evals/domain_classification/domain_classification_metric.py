from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

@numeric_metric(
    name="domain-classification",
    allowed_values=(0.0, 1.0)
)
def domain_classification_metric(prediction: str, actual: str) -> MetricResult:
    """Calculate correctness of the domain classification."""
    return (
        MetricResult(value=1, reason="")
        if prediction == actual
        else MetricResult(value=0, reason="")
    )


def calculate_total_domain_classification_score(experiment_results: Experiment):
    total = len(experiment_results)
    correct = sum(1 for r in experiment_results if r["score"] == 1)
    total_score = correct / total if total > 0 else 0

    return {
        'total_score': total_score,
        'correct': correct,
    }
