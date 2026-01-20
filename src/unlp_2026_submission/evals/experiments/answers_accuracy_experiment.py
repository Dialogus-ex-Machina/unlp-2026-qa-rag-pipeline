from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import SingleAnswerQuestion
from unlp_2026_submission.evals.metrics import answer_accuracy_metric
from unlp_2026_submission.workflow import WorkflowState


@experiment()
async def answers_accuracy_experiment(
        row: SingleAnswerQuestion,
        workflow: CompiledStateGraph
):
    correct_answer = row['correct_answer']

    result: WorkflowState = workflow.invoke(
        input={ 'question': row }
    )

    score = await answer_accuracy_metric.ascore(
        prediction=result['answer'],
        actual=correct_answer,
    )

    experiment_view = {
        'question_id': row['question_id'],
        'question_text': row['question_text'],
        'answers': row['answers'],
        'raw_answer': result.get('raw_answer'),
        'correct_answer': correct_answer,
        "score": score.value,
    }

    return experiment_view
