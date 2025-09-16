#!/usr/bin/env pipenv-shebang
from typing import Tuple
from util.data_split import (
    create_split_name,
    merge_and_validate_cinc_dataset,
    merge_and_validate_ticking_dataset,
    display_split,
    assign_split,
    assign_split_extended,
    assign_split_crossfold,
)
from collections import defaultdict

import os
import logging
import pandas as pd
import click

def parse_reference_paths(reference_paths: str, reference_filenames: str) -> Tuple[list, list]:
    return reference_paths.split(":"), reference_filenames.split(":")

@click.group(context_settings={'show_default': True})
@click.option('--LOG_LEVEL', type=click.Choice(['INFO', 'DEBUG', 'FINE']), default='INFO', help='Debug flag level')
@click.pass_context
def cli(ctx, **kwargs):

    logging.basicConfig(level=getattr(logging, kwargs['log_level'], None))

@cli.command()
@click.option(
    "-I",
    "--input_dir",
    required=True,
    help="Path where the data is stored"
)
@click.option(
    "-O",
    "--output_path",
    default='',
    help="Path where to store the output file, including the name of the file."
)
@click.option(
    "-D",
    "--dataset",
    required=True,
    help="Name of the dataset, can be extended",
)
@click.option(
    "-F",
    "--folds",
    type=int,
    help="How many folds for cross-fold validation."
)
@click.option(
    "-S",
    "--data_split",
    default="0.6:0.2:0.2",
    help="The dataset split to use train:valid:test e.g. (0.6:0.2:0.2)"
)
@click.option(
    "-R",
    "--reference_filename",
    default="REFERENCE.csv",
    help="The name of the reference file."
)
@click.option(
    "-P",
    "--pardir",
    is_flag=True,
    help="Set this flag if the directory provided for the input directory is the databases directory."
)
def split(input_dir, output_path, dataset, folds, data_split, reference_filename, pardir, **kwargs):
    logging.info(f'{kwargs=}')

    print(f'Creating split for the {dataset} dataset')

    patients_excluded = []

    patient_missing_files = defaultdict(list)

    num_removed = 0
    old_len = -1

    cinc_datasets = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

    if pardir:
            input_dir = os.path.dirname(os.path.abspath(input_dir))


    if dataset in cinc_datasets:

        online_appendix_path = os.path.join(input_dir, 'annotations', 'Online_Appendix_training_set.csv')
        reference_sqi_path = os.path.join(input_dir, 'annotations', 'updated', dataset, 'REFERENCE_withSQI.csv')
        reference_path = os.path.join(input_dir, dataset, reference_filename)

        annotations = merge_and_validate_cinc_dataset(
            online_appendix_path,
            reference_path,
            reference_sqi_path,
            dataset
        )

        old_len = len(annotations)

        for patient in annotations['patient']:
            if dataset == 'training-a':
                required_files = [os.path.join(input_dir, 'training-a', f'{patient}.{extension}')
                                for extension in ['hea', 'dat', 'wav']]
            else:
                required_files = [os.path.join(input_dir, dataset, f'{patient}.{extension}')
                                for extension in ['hea', 'wav']]
            if not all(os.path.exists(file) for file in required_files):
                for file in required_files:
                    if not os.path.exists(file):
                        patient_missing_files[patient].append(file)
                patients_excluded.append(patient)

        logging.info(f'{patient_missing_files=}')

        num_removed = len(patient_missing_files)

    elif dataset == 'ticking-heart':

        input_dirs, reference_filenames = parse_reference_paths(input_dir, reference_filename)

        annotations = pd.DataFrame()
        for input_dir, reference_filename in zip(input_dirs, reference_filenames):
            reference_path = os.path.join(input_dir, reference_filename)
            additional_annotations = merge_and_validate_ticking_dataset(reference_path)

            if annotations.size < 1:
                annotations = additional_annotations
            else:
                annotations = pd.concat([annotations, additional_annotations], axis=0, ignore_index=True)

        old_len = len(annotations)

    else:
        raise Exception("Dataset is not supported")


    print('Annotations before split...')
    print(annotations)

    excluded_patients_df = annotations[annotations['patient'].isin(  # type: ignore
        patient_missing_files.keys())].assign(split='ignore')  # type: ignore

    annotations = annotations[~annotations['patient'].isin(  # type: ignore
        patient_missing_files.keys())]  # type: ignore

    new_len = len(annotations)

    assert old_len - new_len == num_removed, f'{(old_len, new_len, num_removed)=}'

    print('Excluding the following from the split...')
    print(excluded_patients_df)

    train = float(data_split.split(":")[0])
    valid = float(data_split.split(":")[1])
    test = float(data_split.split(":")[2])

    if folds is not None:
        if dataset in cinc_datasets:
            annotations, splits = assign_split_crossfold(annotations=annotations,  # type: ignore
                                                        folds=folds,
                                                        stratify_key="diagnosis",
                                                        random_state=None)
        else:
            annotations, splits = assign_split_crossfold(annotations=annotations,  # type: ignore
                                                        folds=folds,
                                                        stratify_key="abnormality",
                                                        random_state=None)
    else:
        if dataset in cinc_datasets:
            annotations, splits = assign_split(annotations=annotations,  # type: ignore
                                                        stratify_key="diagnosis",
                                                        ratios={'train': train, 'valid': valid, 'test': test},
                                                        random_state=None)
        else:
            annotations, splits = assign_split(annotations=annotations,  # type: ignore
                                                        ratios={'train': train, 'valid': valid, 'test': test},
                                                        random_state=None)

    annotations = pd.concat([annotations, excluded_patients_df], axis=0).sort_values(by='patient')  # type: ignore

    if output_path == '':
        split_name = create_split_name()
        output_path = os.path.join('splits', f'{split_name}.csv')

    else:
        split_name = os.path.basename(output_path).removesuffix('.csv')

    print('Annotations after split...')
    display_split(annotations, splits)

    print(f'Saving annotations file to {output_path}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        file.write(f'# Saved as {split_name}\n')
        annotations.to_csv(file, sep=',', index=False)


@cli.command()
@click.option(
    "-I",
    "--input_dirs",
    required=True,
    help="The input splits paths seperated by ':' [path/to/1:path/to/2]"
)
@click.option(
    "-F",
    "--folds",
    required=True,
    help="The input folds seperated by ':', must be synchronised order as input_dirs [fold1:fold2]"
)
@click.option(
    "-O",
    "--out_path",
    required=True,
    help="Where to save the final split file to"
)
def merge_generative_splits(input_dirs, folds, out_path):

    input_dirs = input_dirs.split(":")
    folds = [int(fold) for fold in folds.split(":")]
    assert len(input_dirs) == len(folds), "Mismatch in folds and input dirs"

    max_fold = max(folds)

    output_path = os.path.abspath(out_path)

    dataframes = []

    for idx in range(len(folds)):
        
        fold_num = folds[idx]
        input_dir = os.path.abspath(input_dirs[idx])

        df = pd.read_csv(input_dir)
        for i in range(1, max_fold + 1):
            column_name = f'split{i}' if i > 1 else 'split'
            df[column_name] = 'train' if i == fold_num else None

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    with open(output_path, 'w') as f:
    # Write a comment at the top of the file
        f.write(f"# Merged CSV from files: {', '.join([os.path.basename(f) for f in input_dirs])}\n")
        
        # Save the CSV data
        merged_df.to_csv(f, index=False)


if __name__ == "__main__":
    cli(obj={})
