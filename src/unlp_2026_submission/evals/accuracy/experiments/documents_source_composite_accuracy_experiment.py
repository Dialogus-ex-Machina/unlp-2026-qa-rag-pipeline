from langgraph.graph.state import CompiledStateGraph
from ragas import experiment

from unlp_2026_submission.entities import Question
from unlp_2026_submission.evals.accuracy.metrics import document_source_composite_accuracy_metric
from unlp_2026_submission.workflow.state import QAWorkflowState


@experiment()
async def documents_source_composite_accuracy_experiment(
        question: Question,
        workflow: CompiledStateGraph
):
    result: QAWorkflowState = workflow.invoke(
        input={ 'question': question }
    )

    score = await document_source_composite_accuracy_metric.ascore(
        question=question,
        workflow_result=result
    )

    relevant_documents = result.get("relevant_documents") or []
    relevant_documents_ordered = [
        {"document_id": doc.document_id, "page": doc.page_number}
        for doc in relevant_documents
    ]

    experiment_view = {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "answers": question["answers"],
        "correct_answer": question["correct_answer"],
        "correct_document_id": question["doc_id"],
        "correct_document_page": question["page_num"],
        "n_pages": question.get("n_pages"),
        "question_domain": question.get("domain"),
        "predicted_domain": result.get("predicted_domain"),
        "raw_answer": result.get("raw_answer"),
        "answer": result["answer"],
        "relevant_document_id": result["relevant_document_id"],
        "relevant_document_page_num": result["relevant_document_page_num"],
        "relevant_context": result["relevant_context"],
        "relevant_documents": relevant_documents_ordered,
        "score": score.value,
    }

    return experiment_view
