from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.evals.accuracy.metrics import document_source_page_accuracy_metric
from unlp_2026_submission.workflow.state import QAWorkflowState


@experiment()
async def documents_source_page_accuracy_experiment(
        question: Question,
        workflow: CompiledStateGraph
):
    result: QAWorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await document_source_page_accuracy_metric.ascore(
        question=question,
        workflow_result=result
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'correct_document_id': question['doc_id'],
        'correct_document_page': question['page_num'],
        'raw_answer': result.get('raw_answer'),
        'relevant_document_id': result['relevant_document_id'],
        'relevant_document_page_num': result['relevant_document_page_num'],
        'relevant_context': result['relevant_context'],
        "score": score.value,
    }

    return experiment_view
