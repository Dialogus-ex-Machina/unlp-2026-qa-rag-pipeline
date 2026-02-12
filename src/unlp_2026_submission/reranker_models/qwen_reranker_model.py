import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from unlp_2026_submission.entities import RelevantDocument

from .reranker_model import RerankerModel


class QwenRerankerModel(RerankerModel):
    def __init__(
            self,
            cache_dir: str,
            model_name: str = "Qwen/Qwen3-Reranker-0.6B",
            **model_kwargs: Dict[str, Any]
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side='left',
            **model_kwargs,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        ).eval()


    def rerank(
            self,
            query: str,
            documents: List[RelevantDocument]
    ) -> List[RelevantDocument]:
        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                             query=query, doc=doc)
            return output

        def process_inputs(pairs):
            inputs = self._tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            inputs = self._tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
            for key in inputs:
                inputs[key] = inputs[key].to(self._model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs, **kwargs):
            batch_scores = self._model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

        query_doc_pairs = [[query, doc.text] for doc in documents]

        task = 'Given a web search query, retrieve relevant passages that answer the query'

        pairs = [format_instruction(task, query, doc) for query, doc in query_doc_pairs]

        # Tokenize the input texts
        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)

        results: List[RelevantDocument] = []
        for i, score in enumerate(scores):
            document = documents[i]
            document.relevance_score = score
            results.append(document)

        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return results
