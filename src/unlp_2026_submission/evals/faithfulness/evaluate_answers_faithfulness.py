from langgraph.graph.state import CompiledStateGraph

from .answers_faithfulness_experiment import answers_faithfulness_experiment
from .answer_faithfulness_metric import calculate_total_answers_faithfulness


async def evaluate_answers_faithfulness(
        dataset,
        experiment_name: str,
        workflow: CompiledStateGraph
):
    print(f"Running experiment: {experiment_name}")

    experiment_results = await answers_faithfulness_experiment.arun(
        dataset=dataset,
        name=experiment_name,
        workflow=workflow
    )

    # Calculate accuracy
    answers_faithfulness_result = calculate_total_answers_faithfulness(
        experiment_results
    )

    print(f"\nExperiment {experiment_name} completed!")
    print(f"Total questions: {len(experiment_results)}")
    print(f"Correct answers: {answers_faithfulness_result['correct']}")
    print(f"Answers faithfulness: {answers_faithfulness_result['faithfulness']:.5%}")

    return experiment_results, answers_faithfulness_result['faithfulness']
