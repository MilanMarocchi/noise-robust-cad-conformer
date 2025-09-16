"""
    schedule.py
    Author: Milan Marocchi
    
    Purpose: Contains functions for working with the schedule files
"""

import os
import json

def validate_schedule(schedule: dict) -> dict:
    # Check for the relevant keys
    try: 
        test = schedule["test_set"]
        valid = schedule["valid_set"]

        test["data"]
        valid["data"]
        test["split"]
        valid["split"]
        test["segment"]
        valid["segment"]
        valid["augment_num"]

        datasets = schedule["datasets"]
        for dataset in datasets.values():
            dataset["path"]
            dataset["split"]
            dataset["segment"]
            dataset["gen_data"]
            dataset["augment_num"]

        # This is an optional tag so is not required.
        if "combined_datasets" in schedule.keys():
            combined_datasets = schedule["combined_datasets"]
            for dataset in combined_datasets.values():
                base_sets = dataset["base_sets"]
                proportions = dataset["proportion"]
                augment_num = dataset["augment_num"]

                if augment_num < 0:
                    raise ValueError("Augmentations must be positive.")

                for proportion in proportions:
                    if proportion < 0.0 or proportion > 1.0:
                        raise ValueError("Proportions must be [0, 1]")
                for set in base_sets:
                    if set not in datasets.keys():
                        raise ValueError(f"Base set: {set} does not exist")

        schedule = schedule["schedule"]
        for val in schedule:
            if val["key"] not in datasets.keys() and val["key"] not in combined_datasets.keys():
                raise ValueError(f"Missing key: {val['key']}")

    except ValueError as e:
        raise ValueError("Invalid format for the schedule: " + str(e))

    return schedule


def get_training_datasets(datasets: dict, combined_datasets: dict) -> dict:
    """Gets all the training datasets from the individual and combined datasets"""
    training_datasets = datasets

    for dataset in training_datasets:
        training_datasets[dataset]['combined_dataset'] = False
        training_datasets[dataset]['proportion'] = list()

    for key in combined_datasets:
        dataset = combined_datasets[key]
        training_datasets[key] = {
            'path': [training_datasets[baseset]['path'] for baseset in dataset["base_sets"]],
            'split': [training_datasets[baseset]['split'] for baseset in dataset["base_sets"]],
            'segment': [training_datasets[baseset]['segment'] for baseset in dataset["base_sets"]],
            'gen_data': [training_datasets[baseset]['gen_data'] for baseset in dataset["base_sets"]],
            'combined_dataset': True,
            #'augment_num': [training_datasets[baseset]['augment_num'] for baseset in dataset["base_sets"]], 
            # FIXME: Integrate the changing logic through different variables
            'augment_num': dataset['augment_num'],
            'proportion': dataset["proportion"]
        }

    return training_datasets


def get_schedule(schedule_str: str) -> dict:
    schedule_path = os.path.abspath(schedule_str)

    with open(schedule_path, "r") as json_file:
        schedule = json.load(json_file)

    validate_schedule(schedule)

    return schedule


def get_data_paths(schedule: dict) -> list[str]:
    data_paths: list[str] = list()

    data_paths.append(schedule["test_set"]["data"])
    data_paths.append(schedule["valid_set"]["data"])

    for dataset in schedule["datasets"]:
        data_paths.append(schedule["datasets"][dataset]["path"])

    return data_paths


def get_split_paths(schedule: dict) -> list[str]:
    split_paths: list[str] = list()

    split_paths.append(schedule["test_set"]["split"])
    split_paths.append(schedule["valid_set"]["split"])

    for dataset in schedule["datasets"]:
        split_paths.append(schedule["datasets"][dataset]["split"])

    return split_paths


def get_segment_paths(schedule: dict) -> list[str]:
    segment_paths: list[str] = list()

    segment_paths.append(schedule["test_set"]["segment"])
    segment_paths.append(schedule["valid_set"]["segment"])

    for dataset in schedule["datasets"]:
        segment_paths.append(schedule["datasets"][dataset]["segment"])

    return segment_paths