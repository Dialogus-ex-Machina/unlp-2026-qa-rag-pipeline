from ragas import Dataset

from .faithfulness_datasets import (
    get_faithfulness_full_dataset,
    get_faithfulness_sports_dataset,
    get_faithfulness_medicine_dataset,
    get_faithfulness_other_dataset
)
from .faithfulness_dataset_name import FaithfulnessDatasetName


class FaithfulnessDatasetFactory:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @staticmethod
    def create(
            data_root_dir: str,
            dataset_name: FaithfulnessDatasetName
    ):

        def get_dataset():
            match dataset_name:
                case FaithfulnessDatasetName.FULL:
                    return get_faithfulness_full_dataset(data_root_dir)
                case FaithfulnessDatasetName.SPORT:
                    return get_faithfulness_sports_dataset(data_root_dir)
                case FaithfulnessDatasetName.MEDICINE:
                    return get_faithfulness_medicine_dataset(data_root_dir)
                case _:
                    raise ValueError("Metric not found.")

        dataset = get_dataset()

        return FaithfulnessDatasetFactory(dataset=dataset)

    def get_dataset(self) -> Dataset:
        return self._dataset
