from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.evals.accuracy.metrics import answer_accuracy_metric
from unlp_2026_submission.workflow import WorkflowState


@experiment()
async def answers_accuracy_experiment(
        question: Question,
        workflow: CompiledStateGraph
):
    correct_answer = question['correct_answer']

    result: WorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await answer_accuracy_metric.ascore(
        prediction=result['answer'],
        actual=correct_answer,
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'raw_answer': result.get('raw_answer'),
        'correct_answer': correct_answer,
        "score": score.value,
    }

    return experiment_view
