from unlp_2026_submission.models.reranker_models import RerankerModel
from unlp_2026_submission.workflow.nodes import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

class ModelRerankerNode(BaseNode):
    reranker_model: RerankerModel

    def __init__(
            self,
            reranker_model: RerankerModel,
    ):
        self.reranker_model = reranker_model

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        relevant_documents = state.get('relevant_documents', [])

        if not relevant_documents:
            print('Relevant context not found. Skipping reranking.')
            return {}

        reranked_relevant_documents = self.reranker_model.rerank(
            query=question['question_text'],
            documents=relevant_documents,
        )

        if not reranked_relevant_documents:
            return {}

        print('Reranked relevant documents:', reranked_relevant_documents)

        return {
            'relevant_documents': reranked_relevant_documents,
        }
