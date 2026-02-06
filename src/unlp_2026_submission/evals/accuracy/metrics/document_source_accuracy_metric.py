from ragas import Experiment
from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
from pathlib import Path

from .accuracy_metric_name import AccuracyMetricName


@numeric_metric(
    name=AccuracyMetricName.DOCUMENT_SOURCES.value,
    allowed_values=(0.0, 1.0)
)
def document_source_accuracy_metric(prediction: str, actual: str) -> MetricResult:
    """Calculate accuracy of the prediction."""
    prediction_stem = Path(prediction).stem
    actual_stem = Path(actual).stem

    return (
        MetricResult(value=1, reason="")
        if prediction_stem == actual_stem
        else MetricResult(value=0, reason="")
    )

def calculate_total_documents_source_accuracy(experiment_results: Experiment):
    total = len(experiment_results)
    correct = sum(1 for r in experiment_results if r["score"] == 1)
    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
    }
