from langgraph.graph.state import CompiledStateGraph

from .experiments import answers_accuracy_experiment
from .metrics import calculate_total_answers_accuracy


async def evaluate_answers_accuracy(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await answers_accuracy_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # Calculate accuracy
    answers_accuracy_result = calculate_total_answers_accuracy(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Correct answers: {answers_accuracy_result['correct']}")
    print(f"Answers accuracy: {answers_accuracy_result['accuracy']:.5%}")

    return experiment_results, answers_accuracy_result['accuracy']
