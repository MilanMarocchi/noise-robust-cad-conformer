"""
    testing.py
    Author : Milan Marocchi

    Purpose : To run the classifier to classify heart signals.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from util.classify_stats import RunningBinaryConfusionMatrix
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

def make_pad_collate(
    x_key="input",          # key in each sample dict for the tensor to batch
    y_key="label",          # optional key for labels (set to None if none)
    time_dim=-1,            # which axis is time in x: -1 for [C,T] or [n_mfcc,T]; 0 for [T,C] or [T]
    pad_value=0.0,          # value to pad with
    pad_to=None,            # optionally pad up to a multiple (e.g., product of conv strides)
    max_len=None,           # optionally cap/truncate to this many time steps
    return_lengths=True,
    return_attention_mask=True,
):
    def _collate(batch):
        xs = [b[x_key] for b in batch]
        ys = None if y_key is None else torch.as_tensor([b[y_key] for b in batch])

        # Move time to axis 0 => [T, ...]
        xs_Tfirst = []
        for x in xs:
            if x.dim() == 1:
                xT = x  # [T]
            else:
                if time_dim == -1:
                    xT = x.transpose(0, -1).contiguous()  # [..., T] -> [T, ...]
                elif time_dim == 0:
                    xT = x
                else:
                    raise ValueError("time_dim must be 0 or -1")
            xs_Tfirst.append(xT)

        lengths = torch.as_tensor([xT.shape[0] for xT in xs_Tfirst], dtype=torch.long)
        L = int(lengths.max().item())

        # Optional: round up to a multiple (useful for conv/pooling stacks)
        if pad_to:
            L = int(math.ceil(L / pad_to) * pad_to)
        if max_len is not None:
            L = min(L, max_len)

        padded, masks = [], []
        for xT, l in zip(xs_Tfirst, lengths):
            # Trim if needed
            if xT.shape[0] > L:
                xT = xT[:L]
                l = torch.tensor(L)

            # Pad to L
            pad_len = L - xT.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + xT.shape[1:]
                pad = xT.new_full(pad_shape, pad_value)
                xT = torch.cat([xT, pad], dim=0)

            padded.append(xT)
            if return_attention_mask:
                m = torch.zeros(L, dtype=torch.bool)
                m[:l.item()] = True
                masks.append(m)

        # Stack and move time back to original place
        X = torch.stack([
            p if (p.dim() == 1 or time_dim == 0) else p.transpose(0, -1).contiguous()
            for p in padded
        ], dim=0)  # -> [B, T] or [B, C, T] (if time_dim == -1) or [B, T, C] (if time_dim == 0)

        out = {x_key: X}
        if ys is not None:
            out[y_key] = ys
        if return_lengths:
            out["lengths"] = lengths
        if return_attention_mask:
            out["attention_mask"] = torch.stack(masks, dim=0)  # [B, L] (True = real, False = pad)
        return out
    return _collate

class Tester():

    def __init__(self, model, datasets):
        self.model = model
        self.dataloaders = {key: DataLoader(datasets[key], batch_size=64, num_workers=2) for key in datasets}

    def test(self):
        raise NotImplementedError("Not implemented.")

    def _setup_labels(self, labels):
        # If not already ints split it.
        try:
            labels = [int(x) for x in labels]
        except ValueError:
            labels = [1 if int(x.split('.')[1]) == 1 else 0 for x in labels]
        labels = torch.tensor(labels)

        return labels

class FineTunerPatientTester(Tester):

    def __init__(self, model, datasets):
        super().__init__(model, datasets)

    def classify_patient(self, inputs, labels, fragment_logits):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu_device = torch.device("cpu")

        inputs = inputs.to(device)
        #self.model.to_empty(device=device)
        self.model = self.model.to(device)
        self.model.eval()
        full_labels = [".".join(x.split('/')[-1].split('.')[0:2]) for x in labels]
        labels = self._setup_labels(labels).to(device)

        # Incase not a hf model
        try:
            if self.model.config.model_type in ["mfccconformer", "ast_classifier", "audio-spectrogram-transformer", "whisper_classifier", "multi_input_audio", "wav2vec_classifier", "mfccconformer"]:
                _, logits, _ = self.model(inputs, labels)
            elif self.model.config.model_type in ["energy_conformer", "energy_transformer_mfcc"]:
                inputs = self.model.backbone.preprocess_mfcc_multichannel_batch(inputs)
                logits = self.model(inputs, torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.int64, device=device), labels)
                logits = logits.logits
            else:
                logits = self.model(inputs, labels)
        except:
            logits = self.model(inputs, labels)

        # Apply softmax if not giving hard outputs
        try:
            logits = F.softmax(logits, dim=1)
        except:
            # Instead of floats giving raw classification
            logits = logits

        # Ensure tensor
        if type(logits) != torch.Tensor:
            logits = torch.Tensor(logits)

        # Update fragment_logits
        for idx, label in enumerate(full_labels):
            fragment_logits[label].append(logits[idx].data.to(cpu_device))

    def test(self, threshold: Optional[float]= None, display: bool = True):
        """
        classifies a dataset using a model and criterion.
          model: The model to use (pytorch model)
          dataloaders: The dataloaders to use (expects a dict of dataloaders)
          criterion: Criterion for loss
          device: device to load onto
        """
        # Set to evaluation mode.
        phase = "test"
        threshold = threshold if threshold is not None else 0.0

        runningCM = RunningBinaryConfusionMatrix()

        # Create a default dict which will store all the predictions
        fragment_logits = defaultdict(list)
        patient_logits = defaultdict(list)
        testdataloader = self.dataloaders[phase]
        with torch.no_grad():
            for inputs in testdataloader:
                input_vals = inputs["input_vals"]
                labels = inputs["label"]
                self.classify_patient(input_vals, labels, fragment_logits)

        # Deal with additions for runningCM
        for key in fragment_logits:
            agg_output = sum(fragment_logits[key]) / len(fragment_logits[key])
            # If tensors
            try:
                pred = 0 if agg_output[0] + threshold > agg_output[1] else 1
            # If hard thresholds
            except:
                pred = 0 if agg_output + threshold <= 0.5 else 1
            patient_logits[key] = pred

            label = 1 if int(key.split(".")[1]) == 1 else 0

            runningCM.update(y_true=[label], y_pred=[pred], loss=0)
            logging.debug(f"labels: {label}, predictions: {pred}")

        if display:
            print('Patient Stats:')
            print(f'{phase}')
            print(runningCM.display_stats(aliases=False))
            print(runningCM.base_stats)
            print('Patients Logits:')
            for item in fragment_logits:
                print(item)
                for logit in fragment_logits[item]:
                    print(f"{logit}", end=",")
                print()
                print(f"Final classification: {patient_logits[item]}")
                print()
                print()


        return runningCM.get_stats()['acc'], runningCM.get_stats()

    def roc_curve(self) -> Tuple[list[float], list[float], list[float]]:
        phase = "test"
        self.model.eval()   # Set model to evaluate mode

        # Create a default dict which will store all the predictions
        fragment_logits = defaultdict(list)
        testdataloader = self.dataloaders[phase]
        with torch.no_grad():
            for inputs in testdataloader:
                input_vals = inputs["input_vals"]
                labels = inputs["label"]
                self.classify_patient(input_vals, labels, fragment_logits)

        tprs = []
        fprs = []
        thresholds = [i/50 for i in range(-60,60)]
        for threshold in tqdm(thresholds, ncols=120):
            runningCM = RunningBinaryConfusionMatrix()
            for key in fragment_logits:
                agg_output = sum(fragment_logits[key]) / len(fragment_logits[key])
                pred = 0 if agg_output[0] + threshold > agg_output[1] else 1
                label = 1 if int(key.split(".")[1]) == 1 else 0
                runningCM.update(y_true=[label], y_pred=[pred], loss=0)

            tprs.append(runningCM.get_stats()['tpr'])
            fprs.append(runningCM.get_stats()['fpr'])

        return tprs, fprs, thresholds

class FineTunerEnsemblePatientTester(FineTunerPatientTester):
    def classify_patient(self, inputs, labels, fragment_logits):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu_device = torch.device("cpu")

        inputs = inputs.to(device)
        #self.model.to_empty(device=device)
        full_labels = [".".join(x.split('/')[-1].split('.')[0:2]) for x in labels]
        labels = self._setup_labels(labels).to(device)

        # Repeat for each model in the ensemble
        for model in self.model:
            model = model.to(device)
            model.eval()
            # Incase not a hf model
            try:
                if model.config.model_type in ["mfccconformer", "ast_classifier", "audio-spectrogram-transformer", "whisper_classifier", "multi_input_audio", "wav2vec_classifier", "mfccconformer"]:
                    _, logits, _ = model(inputs, labels)
                elif self.model.config.model_type in ["energy_conformer", "energy_transformer_mfcc"]:
                    inputs = self.model.backbone.preprocess_mfcc_multichannel_batch(inputs)
                    logits = self.model(inputs, torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.int64, device=device), labels)
                else:
                    logits = model(inputs, labels)
            except Exception as e:
                logits = model(inputs, labels)

            # Apply softmax if not giving hard outputs
            try:
                logits = F.softmax(logits, dim=1)
            except:
                # Instead of floats giving raw classification
                logits = logits

            logging.info(f"{logits=}")
            logging.info(f"{logits.shape=}")
            # Ensure tensor
            if type(logits) != torch.Tensor:
                logits = torch.Tensor(logits)

            # Update fragment_logits
            for idx, label in enumerate(full_labels):
                fragment_logits[label].append(logits[idx].data.to(cpu_device))


class FineTunerFragmentTester(Tester):

    def __init__(self, model, datasets):
        super().__init__(model, datasets)

    def classify_fragment(
            self, 
            inputs: torch.Tensor,
            labels: torch.Tensor, 
            runningCM: RunningBinaryConfusionMatrix, 
            threshold: float
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        logging.info(f"{inputs=}")
        logging.info(f"{inputs.shape=}")
        labels = self._setup_labels(labels).to(device)
        self.model = self.model.to(device)
        self.model.eval()
        #self.model = self.model.to_empty(device=device)

        # May not be a hf model
        try:
            if self.model.config.model_type in ["mfccconformer", "ast_classifier", "audio-spectrogram-transformer", "whisper_classifier", "multi_input_audio", "wav2vec_classifier"]:
                logging.info("here")
                _, logits, _ = self.model(inputs, labels)
            elif self.model.config.model_type in ["energy_conformer", "energy_transformer_mfcc"]:
                inputs = self.model.backbone.preprocess_mfcc_multichannel_batch(inputs)
                logits = self.model(inputs, torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.int64, device=device), labels)
                logits = logits.logits
            else:
                logits = self.model(inputs, labels)
        except:
            logits = self.model(inputs, labels)
        try:
            logging.debug(f"classification logits:{logits}")
            loss = F.cross_entropy(logits, labels)
            loss = float(loss.item()*inputs.size(0))
            if abs(threshold) > 1e-6:
                logits = F.softmax(logits, dim=1)
                preds = torch.Tensor([0 if logits[i][0] + threshold > logits[i][1] else 1 for i in range(max(logits.shape))])
            else:
                _, preds = torch.max(logits, dim=1)
        except Exception as e:
            # instead of floats giving raw classification
            print(f"Using hard-threshold for classification")
            loss = 0.0
            # Ensure logits are tensors
            if type(logits) != torch.Tensor:
                logits = torch.Tensor(logits)
            preds = logits


        # statistics
        print(f"labels: {labels.data.to('cpu')}, logits: {logits.to('cpu')}, preds: {preds.to('cpu')}")
        runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=loss)

        return logits, 

    def test(self, threshold: Optional[float] = None, display: bool = True):
        """
        Runs the classifying algorithm on the model with the holdout test data.
        """
        # Set to evaluation mode.
        phase = "test"
        threshold = threshold if threshold is not None else 0.0

        runningCM = RunningBinaryConfusionMatrix()

        with torch.no_grad():
            for inputs in self.dataloaders[phase]:
                input_vals = inputs["input_vals"]
                labels = inputs["label"]
                self.classify_fragment(input_vals, labels, runningCM, threshold)

        if display:
            print('Fragment Stats:')
            print(f'{phase}')
            print(runningCM.display_stats(aliases=False))
            print(runningCM.base_stats)

        return runningCM.get_stats()

    def roc_curve(self) -> Tuple[list[float], list[float], list[float]]:
        """
        Creates the roc curve.
        """
        phase = "test"
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tprs = []
        fprs = []
        thresholds = [i/50 for i in range(-60,60)]
        for threshold in tqdm(thresholds, ncols=120):
            runningCM = RunningBinaryConfusionMatrix()

            with torch.no_grad():
                for inputs in self.dataloaders[phase]:
                    input_vals = inputs["input_vals"]
                    labels = inputs["label"]

                    input_vals = input_vals.to(device)
                    labels = self._setup_labels(labels).to(device)
                    self.model = self.model.to(device)

                    logits = self.model(input_vals, labels)
                    loss = F.cross_entropy(logits, labels)
                    #logits = F.softmax(logits, dim=1)
                    preds = torch.Tensor([0 if logits[i][0] + threshold > logits[i][1] else 1 for i in range(logits.shape[0])])

                    runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*input_vals.size(0)))

            stats = runningCM.get_stats()
            tprs.append(stats['tpr'])
            fprs.append(stats['fpr'])

        return tprs, fprs, thresholds

    def interpretibility(self) -> dict:
        return {}

    def embeddings(self, extract_features: bool = True) -> Tuple[np.ndarray, np.ndarray]: 
        phase = "test"
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        all_features = [] 
        all_labels = []
        with torch.no_grad():
            for inputs in self.dataloaders[phase]:
                input_vals = inputs["input_vals"]
                labels = inputs["label"]

                input_vals = input_vals.to(device)
                labels = self._setup_labels(labels).to(device)
                self.model = self.model.to(device)
                if extract_features:
                    fts = self.model.extract_features(input_vals, labels)
                else:
                    fts = input_vals.flatten(start_dim=1)

                all_features.extend(fts.to("cpu"))
                all_labels.extend(labels.to("cpu")) 

        return np.asarray(all_features), np.asarray(all_labels)


class FineTunerEnsembleFragmentTester(FineTunerFragmentTester):
    def classify_fragment(
            self, 
            inputs: torch.Tensor,
            labels: torch.Tensor, 
            runningCM: RunningBinaryConfusionMatrix, 
            threshold: float
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        logging.info(f"{inputs=}")
        logging.info(f"{inputs.shape=}")
        labels = self._setup_labels(labels).to(device)
        #self.model = self.model.to_empty(device=device)

        # Repeat for each model in the ensemble
        logits = None
        for idx, model in enumerate(self.model):
            # May not be a hf model
            model = model.to(device)
            model.eval()
            try:
                if model.config.model_type in ["mfccconformer", "ast_classifier", "audio-spectrogram-transformer", "whisper_classifier", "multi_input_audio", "wav2vec_classifier"]:
                    logging.info("here")
                    _, current_logits, _ = model(inputs, labels)
                elif model.config.model_type in ["energy_conformer", "energy_transformer_mfcc"]:
                    inputs = self.model.backbone.preprocess_mfcc_multichannel_batch(inputs)
                    current_logits = self.model(inputs, torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.int64, device=device), labels)
                else:
                    current_logits = model(inputs, labels)
            except:
                current_logits = model(inputs, labels)

            if logits is None:
                logits = current_logits
            else:
                # Concatenate logits along the batch dimension
                if isinstance(logits, np.ndarray) and isinstance(current_logits, np.ndarray):
                    logits = np.concatenate([logits, current_logits], axis=0)
                    labels = torch.cat([labels, labels], dim=0)
                else:
                    logits = torch.cat([logits, current_logits], dim=0)
                    labels = torch.cat([labels, labels], dim=0)
        try:
            logging.debug(f"classification logits:{logits}")
            loss = F.cross_entropy(logits, labels)
            loss = float(loss.item()*inputs.size(0))
            if abs(threshold) > 1e-6:
                logits = F.softmax(logits, dim=1)
                preds = torch.Tensor([0 if logits[i][0] + threshold > logits[i][1] else 1 for i in range(max(logits.shape))])
            else:
                _, preds = torch.max(logits, dim=1)
        except Exception as e:
            # instead of floats giving raw classification
            print(f"Using hard-threshold for classification")
            loss = 0.0
            # Ensure logits are tensors
            if type(logits) != torch.Tensor:
                logits = torch.Tensor(logits)
            preds = logits


        # statistics
        print(f"labels: {labels.data.to('cpu')}, logits: {logits.to('cpu')}, preds: {preds.to('cpu')}")
        runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=loss)

        return logits