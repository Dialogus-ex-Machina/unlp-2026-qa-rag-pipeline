from langgraph.graph.state import CompiledStateGraph

from .experiments import documents_source_page_accuracy_experiment
from .metrics import calculate_total_documents_source_page_accuracy

async def evaluate_documents_source_page_accuracy(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await documents_source_page_accuracy_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # Calculate accuracy
    documents_page_accuracy_result = calculate_total_documents_source_page_accuracy(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Total score: {documents_page_accuracy_result['total_score']}")
    print(f"Pages accuracy: {documents_page_accuracy_result['accuracy']:.5%}")

    return experiment_results, documents_page_accuracy_result['accuracy']
