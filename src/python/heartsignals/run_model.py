#!/usr/bin/env pipenv-shebang
"""
    run_model.py
    Author: Milan Marocchi

    Purpose: To train and test models
"""

import copy
from datetime import datetime
import traceback
from typing import Any, Dict, List
import numpy as np
import optuna
import warnings
import logging
import os
import torch
import torchaudio
from tqdm.auto import tqdm

from models.svms.svm import NeuralSVM

logging.getLogger("datasets").setLevel(logging.ERROR) 
# FIXME: Remove this, currently removes ssm warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from heart_datasets.dataset_factory import DatasetFactory
from util.classify_stats import RunningBinaryConfusionMatrix, average_stats
from learners.model_params import (
    get_model_config,
    get_model_config_from_file,
    get_model_training_args,
    get_model_training_args_from_file
)
import util.reproducible # type: ignore
from util.schedule import (
    get_schedule,
    get_training_datasets
)
from processing.transforms import (
    create_audio_data_ss_transforms,
    create_audio_data_transforms,
    create_data_transforms
)
from learners.training import (
    SupervisedTrainer,
    TrainerArguments,
    UnsupervisedTrainer
)
from learners.testing import (
    FineTunerEnsembleFragmentTester,
    FineTunerEnsemblePatientTester,
    FineTunerFragmentTester,
    FineTunerPatientTester,
)
from models.model_factory import (
    ModelFactory,
    get_audio_models,
    get_multi_models,
    get_single_models,
)
from sklearn import svm
from functools import wraps 
from embedlens import (
    start_server,
    BaseModelInterface,
    BaseDatasetInterface,
)

import click
import torch
import os
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("datasets").setLevel(logging.WARNING)

def setup_lora(model):
    """ Setups lora for the model """
    try:
        if model.config.lora:
            if hasattr(model, "models"):
                for submodel in model.models:
                    for name, param in submodel.named_parameters():
                        # Freeze everything that is not LoRA
                        if "lora" not in name.lower():
                            param.requires_grad = False  # Freeze non-LoRA layers
                        else:
                            param.requires_grad = True  # Keep LoRA layers trainable
            else:
                for name, param in model.named_parameters():
                    # Freeze everything that is not LoRA
                    if "lora" not in name.lower():
                        param.requires_grad = False  # Freeze non-LoRA layers
                    else:
                        param.requires_grad = True  # Keep LoRA layers trainable
    except:
        try:
            if model.config.lora_trained:
                for submodel in model.models:
                    for name, param in submodel.named_parameters():
                        # Freeze everything that is not LoRA
                        if "lora" not in name.lower():
                            param.requires_grad = False  # Freeze non-LoRA layers
                        else:
                            param.requires_grad = True  # Keep LoRA layers trainable
        except:
            logging.info("No lora")
            if hasattr(model, "models"):
                for submodel in model.models:
                    for name, param in submodel.named_parameters():
                            param.requires_grad = True  # Keep LoRA layers trainable
                            print(f"{name=}, {param.requires_grad}")
    return model


def parse_schedule(schedule_str):
    """Parse the schedule string."""
    schedule = []
    for item in schedule_str.split(','):
        dataset, epochs, letskip = item.strip().split(':')
        schedule.append((dataset, int(epochs), letskip == '1'))

    return schedule


def parse_model(model_str):
    """
    Parse the models string.
    @returns: composite type, aux type, number of models, if it is an rnn based model
    """
    models = model_str.split(":")

    rnn_models = get_audio_models()
    single_models = get_single_models() 
    multi_models = get_multi_models()

    if len(models) > 1:
        large_type = models[0]
        aux_type = models[2]
        num_models = int(models[1])
        
        if large_type not in multi_models:
            raise Exception(f"Invalid Model string: {model_str}")

    elif len(models) == 1 and models[0] in single_models:
        large_type = None
        aux_type = models[0]
        num_models = 1

    else:
        raise Exception(f"Invalid Model string: {model_str}")

    return large_type, aux_type, num_models, aux_type in rnn_models


def parse_channels(channels):
    return [str(channel) for channel in channels.split(",")] if ',' in channels else [str(channels)]

def setup_labels(labels):
    # If the labels have not been put into ints
    try:
        labels = [int(label) for label in labels]
    except ValueError:
        labels = [1 if int(x.split('.')[1]) == 1 else 0 for x in labels]

    labels = torch.tensor(labels)

    return labels

@click.group(context_settings={'show_default': True})
@click.option('--LOG_LEVEL', type=click.Choice(['INFO', 'DEBUG', 'FINE']), default='INFO', help='Debug flag level')
@click.pass_context
def cli(ctx, **kwargs):

    logging.basicConfig(level=getattr(logging, kwargs['log_level'], None))


def common_training_options(f):
    f = click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].", type=str)(f)
    f = click.option('--processed_data_dir', '-I', required=True, help="Path to the audio/image dir to save processed audio/images.", type=str)(f)
    f = click.option('--output_path', '-O', default='', help="The path to save the model.", type=str)(f)
    f = click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.", type=str)(f)
    f = click.option('--schedule', '-S', default=None, help='The path to the schedule json file', type=str)(f)
    f = click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [adam/adamw/sgd]', type=str)(f)
    f = click.option('--database', '-B', default='training_a', help='The database being used.', type=str)(f)
    f = click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].', type=str)(f)
    f = click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.', type=bool)(f)
    f = click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification', type=int)(f)
    f = click.option('--aux_trained_model_path', '-X', help='The path to a aux pre-trained model.', type=str)(f)
    f = click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in seconds')(f)
    f = click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.', type=bool)(f)
    return f


def common_testing_options(f):
    f = click.option('--data_dir', '-D', required=True, help="The dataset to use if training.", type=str)(f)
    f = click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.", type=str)(f)
    f = click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.", type=str)(f)
    f = click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].", type=str)(f)
    f = click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].", type=str)(f)
    f = click.option('--processed_data_dir', '-I', required=True, help="Path to the image dir to save generated images.", type=str)(f)
    f = click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.", type=str)(f)
    f = click.option('--database', '-B', default='training_a', help='The database being used.', type=str)(f)
    f = click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].', type=str)(f)
    f = click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.', type=bool)(f)
    f = click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification', type=int)(f)
    f = click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')(f)
    f = click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.', type=bool)(f)
    return f


@cli.command()
@common_training_options
@click.option('--channels', '-H', default="3", help="The channels to use if they need to be specified ef (1,2,4)")
@click.option('--noise_cancel', '-N', is_flag=True, help='To apply noise cancelling to the waveform')
@click.option('--folds', default=1, help="Number of folds for crossfold validation")
@click.option('--model_config_file', default=None, help="The path to a model config file to use for training.")
@click.option('--training_args_file', default=None, help="The path to a model training args file to use for training.")
@click.option('--ensemble_models', default=1, help="Number of ensemble models to train.")
def train_audio_model(
        model_str,
        processed_data_dir,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        fs,
        sig_len,
        channels,
        skip_data_valid,
        noise_cancel,
        folds,
        model_config_file,
        training_args_file,
        ensemble_models,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir."""
    del kwargs

    models_factory = None
    datasets = dict()
    large_type, aux_type, num_models, is_rnn = parse_model(model_str)
    channels = parse_channels(channels)
    class_names = ['0', '1']
    audio_dir = processed_data_dir
    best_epoch_measure = -1

    num_channels = 4 if four_bands else 1
    num_channels += 1 if database=="training-a" else 0
    num_channels = (len(channels) * 4 if four_bands else (len(channels) if large_type is None else 1)) if database in ["ticking-heart", "vest-data", "vest-data-matt"] else num_channels

    models_config = get_model_config(
        aux_type, 
        num_channels, 
        num_models, 
        database, 
        len(class_names), 
        fs,
        sig_len,
        large_model=large_type is not None,
        is_ecg='E' in channels
    ) 
    if model_config_file is not None:
        models_config = get_model_config_from_file(
            model_config_file, 
            num_channels, 
            num_models, 
            len(class_names), 
            fs, 
            sig_len, 
    )
    logging.info(f"{models_config=}")

    schedule_dict = get_schedule(schedule)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = create_audio_data_ss_transforms(fs)

    train_datasets = get_training_datasets(schedule_dict["datasets"], schedule_dict["combined_datasets"])
    schedule = [(train_datasets[x["key"]], x["epochs"], x["letskip"]) for x in schedule_dict["schedule"]]
    models_factory = ModelFactory(DEVICE, class_names, freeze=False) 

    best_models_valid_stats = []
    best_models_test_fragment_stats = []
    best_models_test_subject_stats = []
    best_models_ensemble_valid_stats = []
    best_models_ensemble_test_fragment_stats = []
    best_models_ensemble_test_subject_stats = []
    best_models_test_fragment_stats_svm = []
    best_models_test_subject_stats_svm = []
    best_models_ensemble_test_fragment_stats_svm = []
    best_models_ensemble_test_subject_stats_svm = []

    for fold in range(1, folds+1):


        phases = ('valid', 'test')
        datasets = {p: DatasetFactory(
            schedule_dict[f"{p}_set"]["data"],
            schedule_dict[f"{p}_set"]["split"],
            schedule_dict[f"{p}_set"]["segment"],
            p,
            audio_dir,
            database,
            ecg=(database == "training-a"),
            databases=database,
            segmentation=segmentation,
            transform=transforms[p],
            four_band=four_bands,
            channels=channels,
            sig_len=sig_len,
            skip_data_valid=skip_data_valid,
            fs=fs,
            audio_dataset=is_rnn,
            augment_num=(0 if p == 'test' else schedule_dict[f"valid_set"]["augment_num"]),
            fold=fold,
            multihead_dataset=(large_type=="multihead"),
        ).get_dataset() for p in phases}

        idx = 0
        for dataset, num_epochs, letskip in schedule:
            idx += 1

            training_args = get_model_training_args(
                aux_type, 
                num_channels, 
                num_models,
                database, 
                f'{os.path.join(output_path, "dataset"+str(idx), "fold"+str(fold))}',
                num_epochs,
                optimizer,
                large_model=large_type is not None
            )
            if training_args_file is not None:
                training_args = get_model_training_args_from_file(
                    training_args_file, 
                    num_models,
                    f'{os.path.join(output_path, "dataset"+str(idx), "fold"+str(fold))}',
                    num_epochs,
                    optimizer,
                )
            for args in training_args:
                print(f"{args=}")

            # Create the train datasets
            datasets["train"] = DatasetFactory(
                dataset["path"],
                dataset["split"],
                dataset["segment"],
                "train",
                audio_dir,
                database,
                ecg=(database == "training-a"),
                segmentation=segmentation,
                four_band=four_bands,
                sig_len=sig_len,
                transforms=transforms,
                fs=fs,
                channels=channels,
                augment_num=dataset["augment_num"],
                skip_data_valid=skip_data_valid, 
                proportions=dataset["proportion"],
                gens=dataset["gen_data"],
                noise_cancel=noise_cancel,
                generative_dataset=dataset["gen_data"],
                combined_dataset=dataset["combined_dataset"],
                audio_dataset=is_rnn,
                fold=fold,
                multihead_dataset=(large_type=="multihead"),
            ).get_dataset()
            logging.info(f"{datasets=}")

            models = []
            models_svm = []
            for model_num in range(ensemble_models):

                model = None # Resetting model for each fold/model
                aux_models = list() # Resetting model for each fold/model
                best_epoch_measure = -1 # Resetting to not break multidataset logic
                best_epoch_stats = None # Resetting to not break multidataset logic

                if model is None and trained_model_path is None:
                        if large_type is not None:
                            aux_models = [] 
                            for i in range(int(num_models)):
                                aux_model = models_factory.create_model(aux_type, models_config[i])
                                aux_models.append(aux_model)

                        model = models_factory.create_model(
                            large_type if large_type is not None else aux_type, 
                            models_config[-1], 
                            models=aux_models if large_type is not None else None, 
                            aux_model_code=aux_type if large_type is not None else None,
                            freeze_extractor=False
                        )
                        model = setup_lora(model)

                elif model is None and trained_model_path is not None:
                    model_class = models_factory.get_class(large_type if large_type is not None else aux_type)
                    model_config = (model_class.config_class).from_pretrained(os.path.join(trained_model_path, "trained_model")) # type: ignore
                    model = model_class.from_pretrained(os.path.join(trained_model_path, "trained_model"), model_config=model_config, ignore_mismatched_sizes=True) # type: ignore
                    model.to(DEVICE) # type: ignore
                
                if database == "training_a" or (num_channels == 1 and num_models == 1):
                    datasets["train"].channel = 0 # type: ignore
                    datasets["valid"].channel = 0 # type: ignore
                    datasets["test"].channel = 0 # type: ignore
                elif database == "training_a-e":
                    datasets["train"].channel = 1 # type: ignore
                    datasets["valid"].channel = 1 # type: ignore
                    datasets["test"].channel = 1 # type: ignore
                else:
                    datasets["train"].channel = -1 # type: ignore
                    datasets["valid"].channel = -1 # type: ignore
                    datasets["test"].channel = -1 # type: ignore

                trainer = SupervisedTrainer(
                    model=model, # type: ignore
                    args=training_args[-1],
                    train_dataset=datasets["train"],
                    eval_dataset=datasets["valid"]
                )
                model, best_epoch_stats, best_epoch_measure = trainer.train(letskip=letskip, best_epoch_measure=best_epoch_measure, best_stats=best_epoch_stats)
                best_models_valid_stats.append(best_epoch_stats)
                trainer.save_model()

                # Get the test scores or something here
                print()
                fragment_stats = FineTunerFragmentTester(model, datasets).test()
                print()
                for key in datasets:
                    # Turn on evaluate mode
                    datasets[key].evaluate = True # type: ignore
                # Evaluate on the patients here or something
                _, patient_stats = FineTunerPatientTester(model, datasets).test()
                print()
                models.append(model)

                best_models_test_fragment_stats.append(fragment_stats)
                best_models_test_subject_stats.append(patient_stats)

                print(f"Running with SVM instead...\n")
                try:
                    datasets_copy = copy.deepcopy(datasets)
                    svm = NeuralSVM(model).fit(datasets_copy["train"])
                    print()
                    fragment_stats = FineTunerFragmentTester(svm, datasets).test()
                    print()
                    for key in datasets:
                        # Turn on evaluate mode
                        datasets[key].evaluate = True # type: ignore
                    # Evaluate on the patients here or something
                    _, patient_stats = FineTunerPatientTester(svm, datasets).test()
                    print()
                    models_svm.append(svm)

                    best_models_test_fragment_stats_svm.append(fragment_stats)
                    best_models_test_subject_stats_svm.append(patient_stats)
                except Exception as e:
                    print(f"Does not support use of an SVM: {e}")
                    print(f"{traceback.print_exc()}")

            # Test the ensemble model (Just average both the models stats)
            if len(models) > 1:
                # Get the test scores or something here
                print()
                fragment_stats = FineTunerEnsembleFragmentTester(models, datasets).test()
                print()
                for key in datasets:
                    # Turn on evaluate mode
                    datasets[key].evaluate = True # type: ignore
                # Evaluate on the patients here or something
                _, patient_stats = FineTunerEnsemblePatientTester(models, datasets).test()
                print()
                models.append(model)

                best_models_ensemble_test_fragment_stats.append(fragment_stats)
                best_models_ensemble_test_subject_stats.append(patient_stats)

                print(f"Running with SVM instead...\n")
                try:
                    datasets_copy = copy.deepcopy(datasets)
                    svm = NeuralSVM(model).fit(datasets_copy["train"])
                    print()
                    fragment_stats = FineTunerEnsembleFragmentTester(models_svm, datasets).test()
                    print()
                    for key in datasets:
                        # Turn on evaluate mode
                        datasets[key].evaluate = True # type: ignore
                    # Evaluate on the patients here or something
                    _, patient_stats = FineTunerEnsemblePatientTester(models_svm, datasets).test()
                    print()
                    models_svm.append(model)

                    best_models_ensemble_test_fragment_stats_svm.append(fragment_stats)
                    best_models_ensemble_test_subject_stats_svm.append(patient_stats)
                except Exception as e:
                    print(f"Does not support use of an SVM: {e}")
                    print(f"{traceback.print_exc()}")

            # clean up 
            del models
            del models_svm
            del model
            del aux_models

    if ensemble_models > 1:
        best_models_ensemble_test_fragment_stats = average_stats(best_models_ensemble_test_fragment_stats) # type: ignore
        best_models_ensemble_test_subject_stats = average_stats(best_models_ensemble_test_subject_stats) # type: ignore
        best_models_ensemble_test_fragment_stats_svm = average_stats(best_models_ensemble_test_fragment_stats_svm) # type: ignore
        best_models_ensemble_test_subject_stats_svm = average_stats(best_models_ensemble_test_subject_stats_svm) # type: ignore

        print(f"Stats from MLP classifier")
        print(f'{best_models_ensemble_valid_stats=}')
        print(f'{best_models_ensemble_test_fragment_stats=}')
        print(f'{best_models_ensemble_test_subject_stats=}')
        print(f"")
        print(f"Stats from SVM classifier")
        print(f'{best_models_ensemble_test_fragment_stats_svm=}')
        print(f'{best_models_ensemble_test_subject_stats_svm=}')
        print(f"")
    else:
        best_models_valid_stats = average_stats(best_models_valid_stats) # type: ignore
        best_models_test_fragment_stats = average_stats(best_models_test_fragment_stats) # type: ignore
        best_models_test_subject_stats = average_stats(best_models_test_subject_stats) # type: ignore
        best_models_test_fragment_stats_svm = average_stats(best_models_test_fragment_stats_svm) # type: ignore
        best_models_test_subject_stats_svm = average_stats(best_models_test_subject_stats_svm) # type: ignore

        print(f"Stats from MLP classifier")
        print(f'{best_models_valid_stats=}')
        print(f'{best_models_test_fragment_stats=}')
        print(f'{best_models_test_subject_stats=}')
        print(f"")
        print(f"Stats from SVM classifier")
        print(f'{best_models_test_fragment_stats_svm=}')
        print(f'{best_models_test_subject_stats_svm=}')
        print(f"")


@cli.command()
@common_training_options
@click.option('--db_username', default="root", help="The username for the db for the study")
@click.option('--db_password', required=True, help="The password for the db for the study")
@click.option('--db_name', default="optim", help="The name of the db to store the HPO")
@click.option('--study_name', default="optimise", help="The name of the study")
def tune_mfcconformer(
    model_str,
    processed_data_dir,
    output_path,
    schedule,
    database,
    segmentation,
    four_bands,
    fs,
    sig_len,
    skip_data_valid,
    db_username,
    db_password,
    db_name,
    **kwargs
):
    """
    A hardcoded function for now to tune whatever you want
    """
    folds = 5
    noise_cancel = False
    models_factory = None
    datasets = dict()
    sub_model_type = model_str
    is_rnn = True
    class_names = ['0', '1']
    audio_dir = processed_data_dir

    schedule_dict = get_schedule(schedule)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = create_audio_data_transforms(fs)

    train_datasets = get_training_datasets(schedule_dict["datasets"], schedule_dict["combined_datasets"])
    schedule = [(train_datasets[x["key"]], x["epochs"], x["letskip"]) for x in schedule_dict["schedule"]]
    models_factory = ModelFactory(DEVICE, class_names, freeze=False) 

    def objective(trial):
        # Get params/ setup params
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        CHANNELS = ['1', '2', '3', '4']

        optim_config = {
            "output_dir": output_path,
            "batch_size": batch_size,
            "mini_batch": min(batch_size, 64),
            "learning_rate" : trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            "momentum" : trial.suggest_float("momentum", 0.1, 0.99),
            "weight_decay" : trial.suggest_float("weight_decay", 1e-8, 1e-4),
            "step_size" : trial.suggest_int("step_size", 1, 10),
            "gamma" : trial.suggest_float("gamma", 1e-3, 0.4, log=True),
            "optim": trial.suggest_categorical("optim", ["adam", "adamw", "rmsprop"]),
        }

        mlp_hidden_dim = trial.suggest_categorical("mlp_hidden_dim", [128, 256, 512, 1024])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_blocks = trial.suggest_int("num_blocks", 2, 6)
        num_layers = trial.suggest_int("num_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        beta = trial.suggest_float("beta", 0.0, 1.0)
        temperature = trial.suggest_float("temperature", 0.1, 1.0)
        center = trial.suggest_float("center", 0.0001, 0.3, log=True)

        models_config = [{
            "alpha": alpha,
            "beta": beta,
            "temperature": temperature,
            "center": center,
            "hidden_dim": hidden_dim,
            "mlp_hidden_dim": mlp_hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "num_channels": len(CHANNELS),
        }]

        best_models_valid_stats = []
        model_str = "mfcconformer"
        large_type, aux_type, num_models, is_rnn = parse_model(model_str)

        channel_str = ",".join(CHANNELS)
        print(f"{channel_str=}")
        channels = parse_channels(channel_str)
        num_channels = len(CHANNELS)
        # Run through each fold five times to average out randomness
        print(f"Trial {trial.number}, Parameters: {trial.params}")

        for _ in range(5):
            for fold in range(1, folds+1):

                model = None # Resetting model for each fold
                aux_models = list() # Resetting model for each fold
                best_epoch_measure = -1 # Resetting to not break multidataset logic
                best_epoch_stats = None # Resetting to not break multidataset logic

                phases = ('valid', 'test')
                datasets = {p: DatasetFactory(
                    schedule_dict[f"{p}_set"]["data"],
                    schedule_dict[f"{p}_set"]["split"],
                    schedule_dict[f"{p}_set"]["segment"],
                    p,
                    audio_dir,
                    database,
                    ecg=(database == "training-a"),
                    databases=database,
                    segmentation=segmentation,
                    transform=transforms[p],
                    four_band=four_bands,
                    channels=channels,
                    sig_len=sig_len,
                    skip_data_valid=skip_data_valid,
                    fs=fs,
                    audio_dataset=is_rnn,
                    augment_num=(0 if p == 'test' else schedule_dict[f"valid_set"]["augment_num"]),
                    fold=fold,
                    multihead_dataset=(large_type=="multihead"),
                ).get_dataset() for p in phases}

                idx = 0
                for dataset, num_epochs, letskip in schedule:
                    idx += 1

                    optim_config["num_epochs"] = num_epochs

                    # Create the train datasets
                    datasets["train"] = DatasetFactory(
                        dataset["path"],
                        dataset["split"],
                        dataset["segment"],
                        "train",
                        audio_dir,
                        database,
                        ecg=(database == "training-a"),
                        segmentation=segmentation,
                        four_band=four_bands,
                        sig_len=sig_len,
                        transforms=transforms,
                        fs=fs,
                        channels=channels,
                        augment_num=dataset["augment_num"],
                        skip_data_valid=skip_data_valid, 
                        proportions=dataset["proportion"],
                        gens=dataset["gen_data"],
                        noise_cancel=noise_cancel,
                        generative_dataset=dataset["gen_data"],
                        combined_dataset=dataset["combined_dataset"],
                        audio_dataset=is_rnn,
                        fold=fold,
                        multihead_dataset=(large_type=="multihead"),
                    ).get_dataset()
                    logging.info(f"{datasets=}")

                    model = models_factory.create_model(
                        large_type if large_type is not None else aux_type, 
                        models_config[-1], 
                        models=aux_models if large_type is not None else None, 
                        aux_model_code=aux_type if large_type is not None else None,
                        freeze_extractor=False
                    )
                
                    datasets["train"].channel = -1 # type: ignore
                    datasets["valid"].channel = -1 # type: ignore
                    datasets["test"].channel = -1 # type: ignore

                    trainer = SupervisedTrainer(
                        model=model, # type: ignore
                        args=TrainerArguments(**optim_config),
                        train_dataset=datasets["train"],
                        eval_dataset=datasets["valid"]
                    )
                    model, best_epoch_stats, best_epoch_measure = trainer.train(letskip=letskip, best_epoch_measure=best_epoch_measure, best_stats=best_epoch_stats)
                    best_models_valid_stats.append(best_epoch_stats)

        best_models_valid_stats = average_stats(best_models_valid_stats)

        return best_models_valid_stats['mcc']

    study = optuna.create_study(
        direction="maximize",
        study_name=f"optimise_th_{sub_model_type}_ch123456",
        storage=f"mysql://{db_username}:{db_password}@localhost/{db_name}",
        load_if_exists=True,
    )
    study.optimize(objective, catch=Exception, n_trials=250)

    best_trial = study.best_trial
    print(f"Best trial ID: {best_trial.number}")
    print("Best trial hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    cli(obj={})
