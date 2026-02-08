from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.evals.accuracy.metrics import document_source_page_accuracy_metric
from unlp_2026_submission.workflow.state import WorkflowState


@experiment()
async def documents_source_page_accuracy_experiment(
        question: Question,
        workflow: CompiledStateGraph
):
    result: WorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await document_source_page_accuracy_metric.ascore(
        question=question,
        workflow_result=result
    )

    reference_page = result.get('reference_document_page')
    reference_page_text = reference_page.text if reference_page else ""

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'correct_document_id': question['doc_id'],
        'correct_document_page': question['page_num'],
        'raw_answer': result.get('raw_answer'),
        'reference_document_id': result['reference_document_id'],
        'reference_document_page_num': result['reference_document_page_num'],
        'reference_document_page_text': reference_page_text,
        "score": score.value,
    }

    return experiment_view
