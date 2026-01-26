from typing import Any, Callable, Awaitable

from langgraph.graph.state import CompiledStateGraph
from ragas import Dataset

from .metrics import AccuracyMetricName
from .evaluate_composite_accuracy import evaluate_composite_accuracy
from .evaluate_answers_accuracy import evaluate_answers_accuracy
from .evaluate_documents_source_accuracy import evaluate_documents_source_accuracy
from .evaluate_documents_source_page_accuracy import evaluate_documents_source_page_accuracy

class AccuracyEvaluationFactory:
    evaluation_fn: Callable[
        [Dataset, str, CompiledStateGraph],
        Awaitable[tuple]
    ]

    def __init__(self, evaluation_fn: Any):
        self.evaluation_fn = evaluation_fn

    @staticmethod
    def create(metric_name: AccuracyMetricName):
        def get_evaluation_fn():
            match metric_name:
                case AccuracyMetricName.COMPOSITE:
                    return evaluate_composite_accuracy
                case AccuracyMetricName.DOCUMENT_SOURCES:
                    return evaluate_documents_source_accuracy
                case AccuracyMetricName.DOCUMENT_SOURCE_PAGES:
                    return evaluate_documents_source_page_accuracy
                case AccuracyMetricName.ANSWERS:
                    return evaluate_answers_accuracy
                case _:
                    raise ValueError("Metric not found.")

        evaluation_fn = get_evaluation_fn()

        return AccuracyEvaluationFactory(
            evaluation_fn=evaluation_fn
        )

    async def run(
            self,
            dataset: Dataset,
            experiment_name: str,
            workflow: CompiledStateGraph
    ):
        return await self.evaluation_fn(
            dataset,
            experiment_name,
            workflow
        )
