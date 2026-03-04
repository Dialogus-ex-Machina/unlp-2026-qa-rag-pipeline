from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.evals.accuracy.metrics import composite_accuracy_metric
from unlp_2026_submission.rag.qa.state import QAWorkflowState


@experiment()
async def composite_accuracy_experiment(
        question: Question,
        workflow: CompiledStateGraph
):
    result: QAWorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await composite_accuracy_metric.ascore(
        question=question,
        workflow_result=result
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'answers': question['answers'],
        'correct_answer': question['correct_answer'],
        'correct_document_id': question['doc_id'],
        'correct_document_page': question['page_num'],
        'raw_answer': result.get('raw_answer'),
        'answer': result['answer'],
        'relevant_document_id': result['relevant_document_id'],
        'relevant_document_page_num': result['relevant_document_page_num'],
        "score": score.value,
    }

    return experiment_view
