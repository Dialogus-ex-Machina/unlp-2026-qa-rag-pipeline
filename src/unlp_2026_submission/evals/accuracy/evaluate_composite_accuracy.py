from langgraph.graph.state import CompiledStateGraph

from .experiments import composite_accuracy_experiment
from .metrics import calculate_total_composite_accuracy

async def evaluate_composite_accuracy(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await composite_accuracy_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # Calculate accuracy
    composite_accuracy_result = calculate_total_composite_accuracy(
        experiment_results
    )

    print("\nExperiment completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Total score: {composite_accuracy_result['total_score']}")
    print(f"Composite accuracy: {composite_accuracy_result['accuracy']:.5%}")

    return experiment_results, composite_accuracy_result['accuracy']
