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

    results = calculate_total_documents_source_composite_accuracy(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")

    print(f"Documents source composite accuracy: {results['accuracy']:.5%} (total_score: {results['total_score']:.2f})")
    print(f"Document source accuracy: {results['document_source_accuracy']['total']['accuracy']:.5%} (correct: {results['document_source_accuracy']['total']['correct']})")
    print(f"Document page source accuracy: {results['document_page_source_accuracy']['total']['accuracy']:.5%} (total_score: {results['document_page_source_accuracy']['total']['total_score']:.2f})")

    print(f"Recall@1: {results['recall_at_1']:.5%}, Recall@3: {results['recall_at_3']:.5%}, Recall@5: {results['recall_at_5']:.5%}, Recall@10: {results['recall_at_10']:.5%}")

    for domain, metrics in results["document_source_accuracy"]["by_domain"].items():
        print(f"    [{domain}] doc_source: {metrics['accuracy']:.5%}, doc_page: {results['document_page_source_accuracy']['by_domain'][domain]['accuracy']:.5%}")

    return experiment_results, results["accuracy"]
