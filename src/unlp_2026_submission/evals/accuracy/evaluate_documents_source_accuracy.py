from langgraph.graph.state import CompiledStateGraph

from .experiments import documents_source_accuracy_experiment
from .metrics import calculate_total_documents_source_accuracy


async def evaluate_documents_source_accuracy(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await documents_source_accuracy_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # # Calculate accuracy
    documents_accuracy_result = calculate_total_documents_source_accuracy(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Correct sources: {documents_accuracy_result['correct']}")
    print(f"Sources accuracy: {documents_accuracy_result['accuracy']:.5%}")

    return experiment_results, documents_accuracy_result['accuracy']
