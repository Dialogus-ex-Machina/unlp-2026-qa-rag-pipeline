from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.workflow.state import QAWorkflowState
from .domain_classification_metric import domain_classification_metric

@experiment()
async def domain_classification_experiment(
        question: Question,
        workflow: CompiledStateGraph,
):
    correct_domain = question['domain']

    result: QAWorkflowState = workflow.invoke(
        input={ 'question': question }
    )
    predicted_domain = result['predicted_domain']

    score = await domain_classification_metric.ascore(
        prediction=predicted_domain,
        actual=correct_domain
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'predicted_domain': predicted_domain,
        'correct_domain': correct_domain,
        "score": score.value,
    }

    return experiment_view
