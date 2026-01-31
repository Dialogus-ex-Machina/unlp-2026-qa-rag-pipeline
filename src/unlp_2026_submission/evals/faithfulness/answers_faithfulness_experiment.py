from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import QuestionWithContext, DocumentPage
from .answer_faithfulness_metric import answer_faithfulness_metric
from unlp_2026_submission.workflow.state import WorkflowState


@experiment()
async def answers_faithfulness_experiment(
        question: QuestionWithContext,
        workflow: CompiledStateGraph
):
    correct_answer = question['correct_answer']

    reference_document_id = question['doc_id']
    reference_document_page_num = question['page_num']
    reference_document_page = DocumentPage(
        document_id=reference_document_id,
        page_number=reference_document_page_num,
        text=question['doc_text'],
    )

    result: WorkflowState = workflow.invoke(
        input={
            'question': question,
            'reference_document_page': reference_document_page,
            'reference_document_id': reference_document_id,
            'reference_document_page_num': reference_document_page_num,
        }
    )

    score = await answer_faithfulness_metric.ascore(
        prediction=result['answer'],
        actual=correct_answer,
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'document_text': question['doc_text'],
        'document_id': question['doc_id'],
        'document_page_num': question['page_num'],
        'raw_answer': result.get('raw_answer'),
        'correct_answer': correct_answer,
        "score": score.value,
    }

    return experiment_view
