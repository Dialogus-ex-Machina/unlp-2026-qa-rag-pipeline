from unlp_2026_submission.entities import Question
from unlp_2026_submission.rag.index.index_state import IndexState


class SimpleClusterSimilarQuestionNode:
    def __init__(self, overlap_len: int = 250) -> None:
        self.overlap_len = max(0, overlap_len)

    def __call__(self, state: IndexState) -> IndexState:
        questions: list[Question] = state["questions"]

        clusters: dict[str, list[Question]] = {}
        key_order: list[str] = []

        for q in questions:
            doc_text = q.get("doc_text") or ""
            key = doc_text[: self.overlap_len]
            if key not in clusters:
                clusters[key] = []
                key_order.append(key)
            clusters[key].append(q)

        new_questions = [q for key in key_order for q in clusters[key]]

        return {"questions": new_questions}
