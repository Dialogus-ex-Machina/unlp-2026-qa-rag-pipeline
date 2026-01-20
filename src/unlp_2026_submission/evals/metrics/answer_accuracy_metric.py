from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult

@numeric_metric(name="answer_accuracy", allowed_values=(0.0, 1.0))
def answer_accuracy_metric(prediction: str, actual: str) -> MetricResult:
    """Calculate accuracy of the prediction."""
    return (
        MetricResult(value=1, reason="")
        if prediction == actual
        else MetricResult(value=0, reason="")
    )
