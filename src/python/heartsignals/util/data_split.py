"""
Put an thing here 
"""

from typing import Optional
from xkcdpass import xkcd_password 
import datetime
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

def parse_training_a_transducer_dataset(annotations):
    df_filtered = annotations[annotations['Database'] == 'training-a']

    # Create a function to map transducer site to channel
    def map_transducer_to_channel(transducer_site):
        if 'left' in transducer_site.lower() and 'intercoastal' in transducer_site.lower():
            return 4
        elif 'right' in transducer_site.lower() and 'intercoastal' in transducer_site.lower():
            return 6
        elif 'apex' in transducer_site.lower():
            return 2
        elif 'left' in transducer_site.lower() and 'parasternum' in transducer_site.lower():
            return 3
        elif 'right' in transducer_site.lower() and 'parasternum' in transducer_site.lower():
            return 5
        else:
            return None  # Return None for rows to exclude

    # Apply the mapping function to 'Transducer site on body' column
    df_filtered.loc[:, 'Channel'] = df_filtered['Transducer site on body'].apply(map_transducer_to_channel)

    # Drop rows where 'Channel' is None
    df_filtered = df_filtered.dropna(subset=['Channel'])

    # Select only the required columns
    df_final = df_filtered[['Challenge record name', 'Class (-1=normal 1=abnormal)', 'Channel']]
    df_final = df_final.rename(columns={
        'Challenge record name': 'patient', 
        'Class (-1=normal 1=abnormal)': 'abnormality', 
        'Channel': 'channel'
    })

    return df_final 


def parse_training_b_transducer_dataset(annotations):
    df_filtered = annotations[annotations['Database'] == 'training-b']

    # All recordings map to channel 3. 
    df_filtered.loc[:, 'Channel'] = 3

    # Drop rows where 'Channel' is None
    df_filtered = df_filtered.dropna(subset=['Channel'])

    # Select only the required columns
    df_final = df_filtered[['Challenge record name', 'Class (-1=normal 1=abnormal)', 'Channel']]
    df_final = df_final.rename(columns={
        'Challenge record name': 'patient', 
        'Class (-1=normal 1=abnormal)': 'abnormality', 
        'Channel': 'channel'
    })

    return df_final 

def merge_and_validate_cinc_dataset(online_appendix_path, reference_path, reference_sqi_path, database):
    online_appendix = pd.read_csv(online_appendix_path)

    reference = pd.read_csv(reference_path, header=None, names=["patient", "abnormality"])

    reference_sqi = pd.read_csv(reference_sqi_path, header=None, names=["patient", "abnormality", "SQI"])

    filtered_online_appendix = online_appendix[online_appendix['Database'] == database]

    merged_data = pd.merge(filtered_online_appendix, reference,
                           left_on='Challenge record name', right_on='patient', how='inner')

    merged_data = pd.merge(merged_data, reference_sqi, on='patient', how='inner')

    assert all(merged_data['abnormality_x'] == merged_data['abnormality_y']
               ), "Discrepancy found in 'abnormality' values!"

    final_data = merged_data[['patient', 'Diagnosis', 'abnormality_x', 'SQI']]
    final_data.columns = ['patient', 'diagnosis', 'abnormality', 'SQI']

    final_data['diagnosis'] = [s.strip() for s in final_data['diagnosis']]

    return final_data


def merge_and_validate_ticking_dataset(reference_path):

    reference = pd.read_csv(reference_path, header=None, skiprows=1, names=["patient", "recording", "abnormality"])
    # Drop "recording" if it's fully NaN (indicating it wasn’t in the file)
    if reference["abnormality"].isna().all():
        reference = reference.drop(columns=["abnormality"])
        reference = reference.rename(columns={"recording": "abnormality"})  # Rename "recording" to "abnormality"

    return reference


def create_split_name():

    wordlist = xkcd_password.generate_wordlist(xkcd_password.locate_wordfile())
    xkcd_name = xkcd_password.generate_xkcdpassword(wordlist, numwords=3, delimiter='', case='capitalize')

    assert xkcd_name is not None, f'{xkcd_name=}'

    now = datetime.datetime.now()
    mins_name = str(round((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() // 60))

    date_name = now.strftime('%Y-%m%b-%d')

    save_name = '_'.join([
        xkcd_name,
        date_name,
        mins_name,
    ])

    return save_name


def display_split(annotations, splits):

    for split in splits:
        print(annotations[annotations['split'] == split])

    for split in splits:
        print(split, len(splits[split]))
        if 'diagnosis' in splits[split]:
            print(splits[split]['diagnosis'].value_counts())
            print(splits[split]['diagnosis'].value_counts(normalize=True))
        print(splits[split])

    annotations.info()




def random_selection_df(df: pd.DataFrame, proportion: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """Grabs a random selection of data based on the proportion specified."""
    if random_state is not None:
        df = df.sample(frac=proportion, random_state=random_state)
    else:
        df = df.sample(frac=proportion)

    return df

def assign_split(annotations, stratify_key=None, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):

    # Ensure the sum of ratios equals 1
    assert (sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    # Adjust the validation ratio relative to train + valid
    valid_ratio_adjusted = ratios['valid'] / (ratios['train'] + ratios['valid'])

    # Group by patient to ensure splits are done at the patient level
    grouped_patients = annotations.groupby('patient').first().reset_index()

    # Perform patient-level split for the test set
    patients_remaining, patients_test = train_test_split(grouped_patients,
                                                         test_size=ratios['test'],
                                                         random_state=random_state,
                                                         stratify=grouped_patients[stratify_key] if stratify_key is not None else None)

    # Perform patient-level split for train and validation sets
    patients_train, patients_valid = train_test_split(patients_remaining,
                                                      test_size=valid_ratio_adjusted,
                                                      random_state=random_state,
                                                      stratify=patients_remaining[stratify_key] if stratify_key is not None else None)

    # Map patients back to the original annotations
    splits = {'train': annotations[annotations['patient'].isin(patients_train['patient'])],
              'valid': annotations[annotations['patient'].isin(patients_valid['patient'])],
              'test': annotations[annotations['patient'].isin(patients_test['patient'])]}

    # Add a split column to each of the data splits
    splits = {split: splits[split].assign(split=split) for split in splits}

    # Concatenate the splits and sort by patient
    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')

    return new_annotations, splits

def assign_split_crossfold(annotations, folds, stratify_key, random_state=None):

    # Group the annotations by patient and keep a single row per patient
    grouped_patients = annotations.groupby('patient').first().reset_index()

    # Ensure the stratify_key column exists in the grouped patients
    if stratify_key not in grouped_patients.columns:
        raise ValueError(f"The '{stratify_key}' column is required in the grouped patients DataFrame.")

    # Stratified KFold on the grouped patients
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = pd.DataFrame(index=annotations.index)

    # Split based on patient-level information and stratify by the given key
    fold_indices = list(skf.split(grouped_patients, grouped_patients[stratify_key]))

    for fold, (train_val_index, test_index) in enumerate(fold_indices, start=1):
        # The validation fold is the next fold, with wrapping around
        next_fold = (fold % folds)  # Wraps around for cross-validation
        _, valid_index = fold_indices[next_fold]

        # Get the patient IDs for the current fold splits
        patients_train_val = grouped_patients.iloc[train_val_index]['patient'].values
        patients_valid = grouped_patients.iloc[valid_index]['patient'].values
        patients_test = grouped_patients.iloc[test_index]['patient'].values

        # Assign split names for this fold
        split_name = 'split' if fold == 1 else f'split{fold}'
        splits[split_name] = "train"
        splits.loc[annotations['patient'].isin(patients_valid), split_name] = "valid"
        splits.loc[annotations['patient'].isin(patients_test), split_name] = "test"

    # Combine the original annotations with the new splits DataFrame
    new_annotations = pd.concat([annotations, splits], axis=1).sort_values(by='patient')

    return new_annotations, splits

def assign_split_extended(annotations, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):

    assert (sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    valid_ratio_adjusted = ratios['valid'] / (ratios['train'] + ratios['valid'])

    patients_remaining, patients_test = train_test_split(annotations,
                                                         test_size=ratios['test'],
                                                         random_state=random_state,
                                                         stratify=annotations['diagnosis'])

    patients_train, patients_valid = train_test_split(patients_remaining,
                                                      test_size=valid_ratio_adjusted,
                                                      random_state=random_state,
                                                      stratify=patients_remaining['diagnosis'])  # type: ignore

    splits = {'train': patients_train, 'valid': patients_valid, 'test': patients_test}

    splits = {split: splits[split].assign(split=split) for split in splits}  # type: ignore

    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')  # type: ignore

    return new_annotations, splits


def assign_split_noisy_val(annotations, stratify_cols, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):
    assert abs(sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    for col in stratify_cols:
        assert col in annotations.columns, f'{col=} not found in annotations'

    sqi_zero_data = annotations[annotations['SQI'] == 0].copy()
    sqi_nonzero_data = annotations[annotations['SQI'] != 0].copy()

    valid_count_needed = int(len(annotations) * ratios['valid'])
    valid_nonzero_count = valid_count_needed - len(sqi_zero_data)

    non_valid_data, valid_nonzero_data = train_test_split(
        sqi_nonzero_data,
        test_size=valid_nonzero_count,
        random_state=random_state,
        stratify=sqi_nonzero_data[stratify_cols]
    )

    patients_train, patients_test = train_test_split(
        non_valid_data,
        test_size=ratios['test']/(ratios['train'] + ratios['test']),
        random_state=random_state,
        stratify=non_valid_data[stratify_cols]
    )

    patients_valid = pd.concat([sqi_zero_data, valid_nonzero_data])  # type: ignore

    splits = {'train': patients_train, 'valid': patients_valid, 'test': patients_test}

    splits = {split: splits[split].assign(split=split) for split in splits}  # type: ignore

    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')

    return new_annotations, splits


def create_split_name():

    wordlist = xkcd_password.generate_wordlist(xkcd_password.locate_wordfile())
    xkcd_name = xkcd_password.generate_xkcdpassword(wordlist, numwords=3, delimiter='', case='capitalize')

    assert xkcd_name is not None, f'{xkcd_name=}'

    now = datetime.datetime.now()
    mins_name = str(round((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() // 60))

    date_name = now.strftime('%Y-%m%b-%d')

    save_name = '_'.join([
        xkcd_name,
        date_name,
        mins_name,
    ])

    return save_name