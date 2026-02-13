import re
from typing import List, Optional, Tuple
from langchain_core.messages import AIMessage

from unlp_2026_submission.entities import RelevantDocument
from unlp_2026_submission.language_models import LanguageModel
from unlp_2026_submission.workflow.prompts import RerankerPrompt, EngRerankerPrompt

from .reranker_model import RerankerModel


class LLMRerankerModel(RerankerModel):
    top_n: int
    reranker_prompt: RerankerPrompt
    batch_size: int
    language_model: LanguageModel

    def __init__(
        self,
        language_model: Optional[LanguageModel] = None,
        reranker_prompt: RerankerPrompt = None,
        batch_size: int = 10,
        top_n: int = 10,
    ) -> None:
        self.language_model = language_model
        self.reranker_prompt = reranker_prompt or EngRerankerPrompt()
        self.batch_size = batch_size
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        documents: List[RelevantDocument],
    ) -> List[RelevantDocument]:
        if query is None:
            raise ValueError("Query must be provided.")

        if len(documents) == 0:
            return []

        reranked_documents: List[RelevantDocument] = []
        raw_response = None
        for idx in range(0, len(documents), self.batch_size):
            documents_batch = documents[idx : idx + self.batch_size]

            fmt_batch_str = self._format_doc_batch_fn(documents_batch)
            # call each batch independently
            prompt = self.reranker_prompt.format(
                context=fmt_batch_str,
                query=query
            )

            raw_response = self.language_model.invoke(prompt)
            raw_choices, relevances = self._parse_llm_answer_fn(
                raw_response, len(documents_batch)
            )

            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [documents_batch[idx] for idx in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]

            for node, relevance in zip(choice_nodes, relevances):
                node.relevance_score = relevance
                reranked_documents.append(node)

        reranked_documents = sorted(reranked_documents, key=lambda x: x.relevance_score or 0.0, reverse=True)[
            : self.top_n
        ]

        if not reranked_documents:
            print('No relevant documents after rerank.')
            print(
                'Raw rerank response',
                getattr(raw_response, "content", str(raw_response))
            )

        return reranked_documents

    def _format_doc_batch_fn(self, documents: List[RelevantDocument]) -> str:
        """
        Default format node batch function.

        Assign each summary node a number, and format the batch of nodes.

        """
        fmt_node_txts = []
        for idx in range(len(documents)):
            number = idx + 1
            fmt_node_txts.append(
                f"Document {number}:\n"
                f"{documents[idx].text}"
            )
        return "\n\n".join(fmt_node_txts)

    def _parse_llm_answer_fn(
            self,
            response: AIMessage,
            num_choices: int,
            raise_error: bool = False
    ) -> Tuple[List[int], List[float]]:
        """Default parse choice select answer function."""

        answer = getattr(response, "content", str(response))
        answer_lines = answer.split("\n")
        answer_nums = []
        answer_relevances = []
        for answer_line in answer_lines:
            line_tokens = answer_line.split(",")
            if len(line_tokens) != 2:
                if not raise_error:
                    continue
                else:
                    raise ValueError(
                        f"Invalid answer line: {answer_line}. "
                        "Answer line must be of the form: "
                        "answer_num: <int>, answer_relevance: <float>"
                    )
            try:
                answer_num = int(line_tokens[0].split(":")[1].strip())
            except (IndexError, ValueError) as e:
                if not raise_error:
                    continue
                else:
                    raise ValueError(
                        f"Invalid answer line: {answer_line}. "
                        "Answer line must be of the form: "
                        "answer_num: <int>, answer_relevance: <float>"
                    )
            if answer_num > num_choices:
                continue
            answer_nums.append(answer_num)
            # extract just the first digits after the colon.
            try:
                _answer_relevance = re.findall(
                    r"\d+", line_tokens[1].split(":")[1].strip()
                )[0]
                answer_relevances.append(float(_answer_relevance))
            except (IndexError, ValueError) as e:
                if not raise_error:
                    continue
                else:
                    raise ValueError(
                        f"Invalid answer line: {answer_line}. "
                        "Answer line must be of the form: "
                        "answer_num: <int>, answer_relevance: <float>"
                    )
        return answer_nums, answer_relevances
