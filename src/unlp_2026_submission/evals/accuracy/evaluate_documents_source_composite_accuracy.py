from langgraph.graph.state import CompiledStateGraph

from .experiments import documents_source_composite_accuracy_experiment
from .metrics import calculate_total_documents_source_composite_accuracy

async def evaluate_documents_source_composite_accuracy(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await documents_source_composite_accuracy_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # Calculate accuracy
    composite_accuracy_result = calculate_total_documents_source_composite_accuracy(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Total score: {composite_accuracy_result['total_score']}")
    print(f"Documents source composite accuracy: {composite_accuracy_result['accuracy']:.5%}")

    return experiment_results, composite_accuracy_result['accuracy']
