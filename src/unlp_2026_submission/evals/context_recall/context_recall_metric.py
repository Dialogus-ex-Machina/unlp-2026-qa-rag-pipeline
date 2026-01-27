import asyncio
from typing import List
import numpy as np

from ragas import Experiment
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from unlp_2026_submission.language_models import JudgeLanguageModel
from .context_recall_prompt import (
    ContextRecallInput,
    ContextRecallOutput,
    ContextRecallPrompt,
)


class ContextRecall(BaseMetric):
    llm: JudgeLanguageModel

    def __init__(
        self,
        llm: JudgeLanguageModel,
        name: str = "context_recall",
        **kwargs,
    ):
        """
        Initialize ContextRecall metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement classification
            name: The metric name (default: "context_recall")
            **kwargs: Additional arguments passed to BaseMetric
        """
        self.llm = llm
        self.prompt = ContextRecallPrompt()

        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        retrieved_contexts: List[str],
        reference: str,
    ) -> MetricResult:
        """
        Calculate context recall score.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            retrieved_contexts: List of retrieved context strings
            reference: The reference answer to evaluate

        Returns:
            MetricResult with recall score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        await asyncio.sleep(0)

        # Combine contexts into a single string
        context = "\n".join(retrieved_contexts) if retrieved_contexts else ""
        # Create input data and generate prompt
        input_data = ContextRecallInput(
            question=user_input, context=context, answer=reference
        )
        prompt_string = self.prompt.to_string(input_data)
        # Move from old max_tokens param to new
        if (
                self.llm.provider == 'openai' and
                self.llm.model_args is not None and
                self.llm.model_args.get("max_tokens")
        ):
            max_tokens = self.llm.model_args["max_tokens"]
            del self.llm.model_args["max_tokens"]
            self.llm.model_args['max_completion_tokens'] = max_tokens

        # Get classifications from LLM
        result = self.llm.generate(prompt_string, ContextRecallOutput)

        # Calculate score
        if not result.classifications:
            return MetricResult(value=np.nan)

        # Count attributions
        attributions = [c.attributed for c in result.classifications]
        score = sum(attributions) / len(attributions) if attributions else np.nan

        return MetricResult(value=float(score))


def calculate_total_context_recall(
        experiment_results: Experiment
):
    total = len(experiment_results)
    total_score = sum(r["score"] for r in experiment_results)
    recall = total_score / total if total > 0 else 0

    return {
        'recall': recall,
        'total_score': total_score,
    }