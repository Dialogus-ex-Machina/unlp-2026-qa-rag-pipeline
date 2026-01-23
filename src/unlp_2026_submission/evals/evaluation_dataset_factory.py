from ragas import Dataset
from unlp_2026_submission.evals.datasets.datasets import (
    get_full_dataset,
    get_sports_domain_dataset,
    get_medical_domain_dataset
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
                case EvaluationDatasetName.FULL:
                    return get_full_dataset()
                case EvaluationDatasetName.SPORT:
                    return get_sports_domain_dataset()
                case EvaluationDatasetName.MEDICINE:
                    return get_medical_domain_dataset()
                case _:
                    raise ValueError("Metric not found.")

        dataset = get_dataset()

        return EvaluationDatasetFactory(dataset=dataset)

    def get_dataset(self) -> Dataset:
        return self._dataset
