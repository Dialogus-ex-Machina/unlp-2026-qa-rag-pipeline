from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import SingleAnswerQuestion
from unlp_2026_submission.evals.metrics import document_source_accuracy_metric
from unlp_2026_submission.workflow import WorkflowState


@experiment()
async def documents_source_accuracy_experiment(
        question: SingleAnswerQuestion,
        workflow: CompiledStateGraph
):
    correct_document_id = question['doc_id']

    result: WorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await document_source_accuracy_metric.ascore(
        prediction=result['reference_document_id'],
        actual=correct_document_id,
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'correct_document_id': correct_document_id,
        'raw_answer': result.get('raw_answer'),
        'reference_document_id': result['reference_document_id'],
        'reference_document_page': result['reference_document_page'],
        "score": score.value,
    }

    return experiment_view
