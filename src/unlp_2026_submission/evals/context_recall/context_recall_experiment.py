from langgraph.graph.state import CompiledStateGraph
from ragas import experiment
from typing import Optional

from unlp_2026_submission.entities import QuestionWithContext
from unlp_2026_submission.language_models import JudgeLanguageModel
from unlp_2026_submission.workflow.state import WorkflowState
from .context_recall_metric import ContextRecall

def _answer_to_index(choice: str) -> Optional[int]:
    answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    return answer_to_index.get(choice)

@experiment()
async def context_recall_experiment(
        question: QuestionWithContext,
        workflow: CompiledStateGraph,
        judge_language_model: JudgeLanguageModel
):
    correct_answer = question['correct_answer']
    question_text = question['question_text']

    result: WorkflowState = workflow.invoke(
        input={ 'question': question }
    )
    reference_page = result.get('reference_document_page')
    retrieved_contexts = [reference_page.text] if reference_page else [""]
    correct_answer_text = question['answers'][_answer_to_index(correct_answer)]

    scorer = ContextRecall(llm=judge_language_model)
    score = await scorer.ascore(
        user_input=question_text,
        retrieved_contexts=retrieved_contexts,
        reference=correct_answer_text
    )

    experiment_view = {
        'question_id': question['question_id'],
        'question_text': question['question_text'],
        'retrieved_contexts': retrieved_contexts,
        'correct_answer': correct_answer,
        "score": score.value,
    }

    return experiment_view
