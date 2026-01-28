from ragas import Dataset
from unlp_2026_submission.config import Config

from .faithfulness_datasets import (
    get_faithfulness_full_dataset,
    get_faithfulness_sports_dataset,
    get_faithfulness_medical_dataset
)
from .faithfulness_dataset_name import FaithfulnessDatasetName


class FaithfulnessDatasetFactory:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @staticmethod
    def create(
            config: Config,
            dataset_name: FaithfulnessDatasetName
    ):

        def get_dataset():
            match dataset_name:
                case FaithfulnessDatasetName.FULL:
                    return get_faithfulness_full_dataset(config)
                case FaithfulnessDatasetName.SPORT:
                    return get_faithfulness_sports_dataset(config)
                case FaithfulnessDatasetName.MEDICINE:
                    return get_faithfulness_medical_dataset(config)
                case _:
                    raise ValueError("Metric not found.")

        dataset = get_dataset()

        return FaithfulnessDatasetFactory(dataset=dataset)

    def get_dataset(self) -> Dataset:
        return self._dataset
