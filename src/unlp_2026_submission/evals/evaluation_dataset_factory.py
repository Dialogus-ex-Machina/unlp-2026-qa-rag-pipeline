from ragas import Dataset
from unlp_2026_submission.evals.datasets import (
    get_sports_qa_with_context_dataset,
    get_sports_qa_dataset,
    get_full_qa_with_context_dataset,
    get_medical_qa_with_context_dataset,
    get_full_qa_dataset,
    get_medical_qa_dataset
)
from unlp_2026_submission.evals.evaluation_dataset_name import EvaluationDatasetName

class EvaluationDatasetFactory:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @staticmethod
    def create(dataset_name: EvaluationDatasetName):

        def get_dataset():
            match dataset_name:
                case EvaluationDatasetName.FULL_QA:
                    return get_full_qa_dataset()
                case EvaluationDatasetName.SPORT_QA:
                    return get_sports_qa_dataset()
                case EvaluationDatasetName.MEDICINE_QA:
                    return get_medical_qa_dataset()
                case EvaluationDatasetName.FULL_QA_WITH_CONTEXT:
                    return get_full_qa_with_context_dataset()
                case EvaluationDatasetName.MEDICINE_QA_WITH_CONTEXT:
                    return get_medical_qa_with_context_dataset()
                case EvaluationDatasetName.SPORT_QA_WITH_CONTEXT:
                    return get_sports_qa_with_context_dataset()
                case _:
                    raise ValueError("Metric not found.")

        dataset = get_dataset()

        return EvaluationDatasetFactory(dataset=dataset)

    def get_dataset(self) -> Dataset:
        return self._dataset
