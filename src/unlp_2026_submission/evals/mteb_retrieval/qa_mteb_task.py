from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

from .qa_mteb_datasets import get_qa_mteb_dataset

_EVAL_SPLIT = "test"
_LANGUAGES = {
    "ukr": ["ukr-Cyrl"],
}

class QARetrievalTask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CustomQARetrieval",
        dataset={"path": "", 'revision': ""},
        description="Custom QA task",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        prompt="Retrieve text based on user query.",
        main_score="ndcg_at_10",
        domains=["Medical", "Constructed"],
        sample_creation="created",
        task_subtypes=["Question answering"],
        annotations_creators="expert-annotated",
        dialect=[],
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return

        raw_dataset = get_qa_mteb_dataset()
        queries_ds = raw_dataset['queries']
        corpus_ds = raw_dataset['corpus']
        qrels_ds = raw_dataset['qrels']

        qrels_ds = qrels_ds.to_polars()
        qrels_dict = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"]))
            for query_id, group in qrels_ds.group_by("query-id", maintain_order=False)
        }

        self.dataset = {
            'ukr': {
                _EVAL_SPLIT: RetrievalSplitData(
                    corpus=corpus_ds,
                    queries=queries_ds,
                    relevant_docs=qrels_dict,
                    top_ranked=None,
                )
            }
        }

        self.data_loaded = True