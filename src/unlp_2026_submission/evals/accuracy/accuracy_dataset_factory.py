from ragas import Dataset
from .accuracy_datasets import (
    get_accuracy_full_dataset,
    get_accuracy_sports_dataset,
    get_accuracy_medicine_dataset,
)
from .accuracy_dataset_name import AccuracyDatasetName


class AccuracyDatasetFactory:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @staticmethod
    def create(
            data_root_dir: str,
            dataset_name: AccuracyDatasetName
    ):

        def get_dataset():
            match dataset_name:
                case AccuracyDatasetName.FULL:
                    return get_accuracy_full_dataset(data_root_dir)
                case AccuracyDatasetName.SPORT:
                    return get_accuracy_sports_dataset(data_root_dir)
                case AccuracyDatasetName.MEDICINE:
                    return get_accuracy_medicine_dataset(data_root_dir)
                case _:
                    raise ValueError("Metric not found.")

        dataset = get_dataset()

        return AccuracyDatasetFactory(dataset=dataset)

    def get_dataset(self) -> Dataset:
        return self._dataset
