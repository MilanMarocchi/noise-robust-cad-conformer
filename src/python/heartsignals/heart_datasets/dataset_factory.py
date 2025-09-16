"""
    dataset_factory.py
    Author: Milan Marocchi

    Purpose: To generically create the correct dataset
"""

import os
from typing import Tuple

from .vest_data import FeatureVectorsDataset_noWav
from torch.utils.data import Dataset

class DatasetFactory():

    def __init__(
        self,
        data_dir: str,
        split_path: str,
        segment_dir: str,
        data_type: str,
        processed_data_dir: str,
        database: str,
        generative_dataset: bool = False,
        combined_dataset: bool = False,
        audio_dataset: bool = False,
        cinc_dataset: bool = False,
        multihead_dataset: bool = False,
        ensemble_dataset: bool = False,
        **kwargs
    ):
        self.data_dir = data_dir
        self.split_path = split_path
        self.segment_dir = segment_dir
        self.data_type = data_type
        self.processed_data_dir = processed_data_dir
        self.database = database
        self.generative_dataset = generative_dataset
        self.combined_dataset = combined_dataset
        self.audio_dataset = audio_dataset
        self.cinc_dataset = cinc_dataset
        self.multihead_dataset = multihead_dataset
        self.ensemble_dataset = ensemble_dataset

        self.params = kwargs

    def get_dataset(self) -> Dataset:
        dataset_class = self.get_dataset_class(
            self.database,
            self.generative_dataset, 
            self.combined_dataset, 
            self.audio_dataset,
            self.cinc_dataset,
            self.multihead_dataset,
            self.ensemble_dataset,
        ) 

        return dataset_class(
            self.data_dir,
            self.split_path,
            self.segment_dir,
            self.data_type,
            self.processed_data_dir,
            **self.params
        )

    def get_multi_dataset(self, num_models: int) -> Tuple[list[Dataset], Dataset]:
        dataset_class, multi_dataset_class = self.get_multi_datasets_class(
            self.database,
            self.generative_dataset,
            self.combined_dataset,
            self.multihead_dataset,
            self.ensemble_dataset,
        )

        multi_dataset = multi_dataset_class(
            self.data_dir,
            self.split_path,
            self.segment_dir,
            self.data_type,
            self.processed_data_dir,
            **self.params
        )

        # Turn off ecg as only required for large dataset
        dataset_params = self.params.copy()
        dataset_params["ecg"] = False

        datasets = []
        for i in range(int(num_models)):
            dataset = dataset_class(
                self.data_dir,
                self.split_path,
                self.segment_dir,
                self.data_type,
                os.path.join(self.processed_data_dir, str(i)),
                **dataset_params
            )
            datasets.append(dataset)

        return datasets, multi_dataset
        

    def get_dataset_class(self, database, is_gen=False, is_combined=False, is_rnn=False, is_cinc=False, is_multihead=False, is_ensemble=False):
        if database == 'vest-data-matt':
            Dataset = FeatureVectorsDataset_noWav
        else:
            raise ValueError(f"Dataset {database} not recognized")

        return Dataset

    def get_multi_datasets_class(self, database, is_gen=False, is_combined=False, is_multihead=False, is_ensemble=False):
        raise NotImplementedError("No multi datasets implemented yet")