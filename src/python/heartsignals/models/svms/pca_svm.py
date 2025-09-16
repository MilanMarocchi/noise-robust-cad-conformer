import numpy as np
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
import logging
import librosa.feature
import torch.nn as nn
from sklearn import svm
from sklearn.decomposition import PCA
from typing import List, Optional
from torch.utils.data import (
    Dataset,
    DataLoader
)

from processing.filtering import minmax_normalise_signal
from processing.process import normalise_array_length

class PCASVMConfig(PretrainedConfig):
    """
    This is the config class for the custom Wav2Vec Based Audio classifier model 

    Args:
        num_classes (int) : Number of classes to classify
        hidden_size (int) : Size of the hidden layer
    """

    model_type = "pca_svm_classifier"

    def __init__(
        self,
        num_classes=2,
        ft_dim=512,
        sig_len=4,
        fs=4000,
        feature_type="MFCC",
        num_channels=4,
        classifier_layer=0,
        **kwargs
    ):
        self.num_classes = num_classes
        self.ft_dim = ft_dim
        self.sig_len = sig_len
        self.fs = fs
        self.feature_type = feature_type
        self.num_channels = 4
        self.classifier_layer = classifier_layer

        self.num_samples = int(self.fs * self.sig_len) * self.num_channels


class PCASVM(PreTrainedModel):

    config_class = PCASVMConfig 

    def __init__(self, config: PretrainedConfig, model: Optional[PreTrainedModel] = None, **kwargs): 
        super().__init__(config)

        self.svm = None
        self.pca = None
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if self.model is None else self.model.criterion

        self.classifier_layer = self.config.classifier_layer
        self.activations = []
        self.last_activation = None

    def preprocess_mfcc(self,
        waveform, 
        sample_rate=4000, 
        n_mels=21, 
        n_mfcc=13, 
        n_fft=512, 
        hop_length=160, 
        win_length=512, 
        f_min=20, 
        f_max=500
    ):
        # Ensure min-max scaling
        waveform = minmax_normalise_signal(waveform)

        # Compute mfccs
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=f_min,
            fmax=f_max,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )

        # z-normlalise mfccs
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

        return mfcc.reshape(mfcc.shape[0], -1)

    def preprocess_signal(self, input_vals):
        batch_size = input_vals.shape[0]
        input_vals = input_vals.reshape(batch_size, self.config.num_samples)
        input_vals = input_vals.detach().cpu().numpy()

        if self.config.feature_type == "MFCC":
            input_vals = self.preprocess_mfcc(
                input_vals, 
                sample_rate=self.config.fs
            )

        return input_vals

    def forward(self, input_vals, labels, **kwargs):
        # Without altering the feature extraction.
        # with torch.no_grad():)
        if self.svm is None or self.pca is None:
            raise ValueError("Need to first fit the model to data.")

        if self.model is not None:
            hook_handle = self.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)
            self.model(input_vals.to(self.model.device), labels)

        input_vals = self.preprocess_signal(input_vals) if self.model is None else self.last_activation
        pca_signal = self.pca.transform(input_vals)
        out = self.svm.predict(pca_signal)

        if self.model is not None:
            hook_handle.remove()

        return torch.tensor(out)

    def fit(self, datasets: Dataset | List[Dataset]):
        
        dataloaders = [DataLoader(dataset, batch_size=32, shuffle=False) for dataset in datasets]
        self.activations = []

        if self.model is not None:
            try:
                hook_handle = self.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)
            except:
                hook_handle = self.model.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)

        signals = []
        labels = []
        for dataloader in dataloaders:
            for batch in dataloader:
                input_vals = batch["input_vals"]
                label = batch["label"]
                label = [int(1 if i.split('.')[1] == '1' else 0) for i in label] if self.model is not None else label

                if self.model is not None:
                    self.model(input_vals.to(self.model.device), label)
                else:
                    input_vals = self.preprocess_signal(input_vals)
                    signals.append(input_vals)
                labels.extend(label)
        signals = np.concatenate(signals, axis=0) if self.model is None else np.asarray(self.activations)
        labels = np.asarray(labels)

        if self.model is not None:
            hook_handle.remove()

        self.svm = svm.SVC()
        self.pca = PCA(n_components=self.config.ft_dim)
        pca_signal = self.pca.fit_transform(signals)
        self.svm.fit(pca_signal, labels)

        return self

    def _save_activations(self, module, inputs, output):
        activation = inputs[0]
        self.activations.extend(activation.cpu().detach().numpy())
        self.last_activation = activation.cpu().detach().numpy()


