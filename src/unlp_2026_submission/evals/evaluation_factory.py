from typing import Any, Callable, Awaitable

from langgraph.graph.state import CompiledStateGraph
from ragas import Dataset

from .evaluate_answers_faithfulness import evaluate_answers_faithfulness
from .evaluation_metric_name import EvaluationMetricName
from .evaluate_composite_accuracy import evaluate_composite_accuracy
from .evaluate_answers_accuracy import evaluate_answers_accuracy
from .evaluate_documents_source_accuracy import evaluate_documents_source_accuracy
from .evaluate_documents_source_page_accuracy import evaluate_documents_source_page_accuracy

class EvaluationFactory:
    evaluation_fn: Callable[
        [Dataset, str, CompiledStateGraph],
        Awaitable[tuple]
    ]

    def __init__(self, evaluation_fn: Any):
        self.evaluation_fn = evaluation_fn

    @staticmethod
    def create(metric_name: EvaluationMetricName):
        def get_evaluation_fn():
            match metric_name:
                case EvaluationMetricName.COMPOSITE_ACCURACY:
                    return evaluate_composite_accuracy
                case EvaluationMetricName.DOCUMENT_SOURCE_ACCURACY:
                    return evaluate_documents_source_accuracy
                case EvaluationMetricName.DOCUMENT_SOURCE_PAGE_ACCURACY:
                    return evaluate_documents_source_page_accuracy
                case EvaluationMetricName.ANSWER_ACCURACY:
                    return evaluate_answers_accuracy
                case EvaluationMetricName.ANSWER_FAITHFULNESS:
                    return evaluate_answers_faithfulness
                case _:
                    raise ValueError("Metric not found.")

        evaluation_fn = get_evaluation_fn()

        return EvaluationFactory(
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
