from langgraph.graph.state import CompiledStateGraph

from .domain_classification_experiment import domain_classification_experiment
from .domain_classification_metric import calculate_total_domain_classification_score

async def evaluate_domain_classification(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph,
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await domain_classification_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow,
    )

    # Calculate score
    domain_classification_result = calculate_total_domain_classification_score(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Correct questions: {domain_classification_result['correct']}")
    print(f"Context recall: {domain_classification_result['total_score']:.5%}")

    return experiment_results, domain_classification_result['total_score']
