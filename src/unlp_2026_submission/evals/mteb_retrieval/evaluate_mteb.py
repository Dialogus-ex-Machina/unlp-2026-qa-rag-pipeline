import mteb
from mteb.tasks import BelebeleRetrieval, WebFAQRetrieval

from unlp_2026_submission.config import Config
from .mteb_task_name import MTEBTaskName
from .qa_mteb_task import QARetrievalTask


def evaluate_mteb(
        config: Config,
        task_name: MTEBTaskName,
):
    model = mteb.get_model(
        config.embeddings_model_name,
    )

    tasks = []

    if task_name == MTEBTaskName.ALL:
        tasks=[BelebeleRetrieval(), WebFAQRetrieval(), QARetrievalTask()]
    elif task_name == MTEBTaskName.BELEBELE:
        tasks=[BelebeleRetrieval()]
    elif task_name == MTEBTaskName.WEB_FAQ:
        tasks=[WebFAQRetrieval()]
    else:
        tasks=[QARetrievalTask()]

    for task in tasks:
        task.filter_languages(
            languages=["ukr"],
            exclusive_language_filter=True
        )
        task.metadata.main_score = 'ndcg_at_3'

    evaluation_result = mteb.evaluate(
        model,
        tasks=tasks,
        overwrite_strategy="always",
        encode_kwargs={"batch_size": 6}
    )

    return evaluation_result