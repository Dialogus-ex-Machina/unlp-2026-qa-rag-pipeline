from langgraph.graph.state import CompiledStateGraph

from unlp_2026_submission.evals.experiments import answers_accuracy_experiment

async def evaluate_accuracy(
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
    total = len(experiment_results)
    correct = sum(1 for r in experiment_results if r["score"] == 1)
    answers_accuracy = correct / total if total > 0 else 0

    print("\nExperiment completed!")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Answers accuracy: {answers_accuracy:.2%}")

    return experiment_results, answers_accuracy
