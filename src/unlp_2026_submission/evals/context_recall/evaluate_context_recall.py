from langgraph.graph.state import CompiledStateGraph

from unlp_2026_submission.language_models import JudgeLanguageModel
from .context_recall_experiment import context_recall_experiment
from .context_recall_metric import calculate_total_context_recall

async def evaluate_context_recall(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph,
        judge_language_model: JudgeLanguageModel
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await context_recall_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow,
        judge_language_model=judge_language_model
    )

    # Calculate racall
    context_recall_result = calculate_total_context_recall(
        experiment_results
    )

    print("\nExperiment completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Total score: {context_recall_result['total_score']}")
    print(f"Context recall: {context_recall_result['recall']:.5%}")

    return experiment_results, context_recall_result['recall']
