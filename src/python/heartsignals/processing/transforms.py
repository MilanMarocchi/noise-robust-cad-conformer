"""
    transforms.py
    Author: Milan Marocchi
    
    Purpose: Create transforms for ml.
"""

import torch
from processing.augmentation import (
    augment_ecg, 
    augment_pcg, 
    augment_signals, 
    augment_synchronised_pcg
)
from processing.filtering import (
    band_stop,
    time_stretch_crop,
)

from torchvision import transforms
from torchvision.transforms import Lambda
from torch import Tensor
import random
import numpy as np

from processing.process import pre_filter_pcg


class RandomStretch:
    """Applies a random stretch to a signal"""

    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        if random.random() > 0.8:

            stretch_factor = 0.96 + random.random() * (1.04 - 0.96)
            audio = time_stretch_crop(audio, self.fs, stretch_factor)

        return audio
    

class RandomStretchSynchronized:
    """Applies a random stretch to a signal synchronsied with a second signal"""

    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, audio: tuple):
        audio1, audio2 = audio

        if random.random() > 0.8:

            stretch_factor = 0.96 + random.random() * (1.04 - 0.96)
            audio1 = time_stretch_crop(audio1, self.fs, stretch_factor)
            audio2 = time_stretch_crop(audio2, self.fs, stretch_factor)

        return audio1, audio2

class RandomTimeFreqMask:
    """Applies a random line mask in the spectrogram of the audio"""
    def __init__(self, thickness: float, fs: int):
        self.thickness = thickness
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        sig_len = len(audio) 
        assert sig_len > 1000, "Correct way to get sig len for multi-channel"

        time_thickness = int(self.thickness * sig_len)
        freq_thickness = int(self.thickness * self.fs)

        if random.random() > 0.8:
            if random.random() > 0.5:
                # Time masking
                time = random.randint(0, len(audio) - time_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio[time:time + time_thickness] = 0
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1)
                    audio[time:time + time_thickness, channel]
            else:
                # Frequency masking
                frequency = random.randint(1, self.fs - freq_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio = band_stop(audio, self.fs, frequency, frequency + freq_thickness)
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1) 
                    audio[:, channel] = band_stop(audio[:, channel], self.fs, frequency, frequency + freq_thickness)

        return audio

class RandomTimeFreqMaskSynchronised:
    """Applies a random line mask in the spectrogram of the audio"""
    def __init__(self, thickness: float, fs: int):
        self.thickness = thickness
        self.fs = fs

    def __call__(self, audio: tuple):
        audio1, audio2 = audio
        sig_len = len(audio1) 
        assert sig_len > 1000, "Correct way to get sig len for multi-channel"

        time_thickness = int(self.thickness * sig_len)
        freq_thickness = int(self.thickness * self.fs)

        if random.random() > 0.8:
            if random.random() > 0.5:
                # Time masking
                time = random.randint(0, len(audio1) - time_thickness - 1)

                audio1[time:time + time_thickness] = 0
                audio2[time:time + time_thickness] = 0
            else:
                # Frequency masking
                frequency = random.randint(1, self.fs - freq_thickness - 1)

                audio1 = band_stop(audio1, self.fs, frequency, frequency + freq_thickness)
                audio2 = band_stop(audio2, self.fs, frequency, frequency + freq_thickness)

        return audio1, audio2

class RandomLineMask:
    """Applies a random line mask"""
    def __init__(self, line_thickness: int = 10):
        self.line_thickness = line_thickness 

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() > 0.7:
            if random.random() > 0.5:
                # Time masking
                col = random.randint(0, img.shape[2] - 1 - self.line_thickness)
                img[:, :, col:col+self.line_thickness] = 0
            else:
                # Frequency masking
                row = random.randint(0, img.shape[1] - 1 - self.line_thickness)
                img[:, :, row:row+self.line_thickness] = 0

        return img

class RandomPCGAugment:
    """Applies random augmentation to the PCG"""
    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, pcg: np.ndarray):
        if random.random() > 0.75:
            pcg = augment_pcg(pcg, self.fs, prob_hpss=0.25)

class RandomPCGAugmentSynchronised:
    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, pcgs: tuple):
        pcg, pcg_ref = pcgs
        if random.random() > 0.75:
            pcg, pcg_ref = augment_synchronised_pcg(pcg, pcg_ref, self.fs)

        pcg = pre_filter_pcg(pcg, self.fs)
        pcg_ref = pre_filter_pcg(pcg_ref, self.fs)

        return pcg, pcg_ref

class RandomECGAugment:
    """Applies random augmentation to the ECG"""
    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, ecg: np.ndarray):
        if random.random() > 0.75:
            ecg = augment_ecg(ecg, self.fs)
        return ecg

class RandomEPCGAugment:
    """Applies random augmentation to the PCG and ECG"""
    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, epcg: tuple[np.ndarray, np.ndarray]):
        ecg, pcg = epcg
        if random.random() > 0.75:
            ecg, pcg = augment_signals(ecg, pcg, self.fs)

        return ecg, pcg


def numpy_to_tensor(y):
    return torch.from_numpy(y.copy())


def numpy_to_tensor_tuple(y: tuple):
    y1, y2 = y
    return torch.from_numpy(y1.copy()), torch.from_numpy(y2.copy())


def get_pil_transform_numpy(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai but outputting a numpy image
    """

    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.numpy()).astype(np.float64))  # Convert tensor to np array
    ])

    return transf


def get_pil_transform(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return transf


def get_normalise_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def get_preprocess_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def create_data_transforms(is_inception: bool = False) -> dict[str, transforms.Compose]:
    """
    Creates data transforms for training and classifying
    """
    size = 299 if is_inception else 224
    line_thickness = 15 if is_inception else 10

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomLineMask(line_thickness),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def create_audio_data_transforms(fs: int) -> dict[str, transforms.Compose]:

    thickness = 0.09

    data_transforms = {
        'train': transforms.Compose([
            numpy_to_tensor,
            RandomStretch(fs),
            RandomTimeFreqMask(thickness, fs), # type: ignore,
            RandomTimeFreqMask(thickness / 2, fs), # type: ignore
        ]),
        'valid': transforms.Compose([
            numpy_to_tensor,
        ]),
        'test': transforms.Compose([
            numpy_to_tensor,
        ])
    }

    return data_transforms

def create_audio_data_ss_transforms(fs: int) -> dict[str, transforms.Compose]:

    thickness = 0.09

    data_transforms = {
        'train': transforms.Compose([
            RandomPCGAugment(fs),
            RandomStretch(fs),
            RandomTimeFreqMask(thickness, fs), # type: ignore,
            RandomTimeFreqMask(thickness / 2, fs), # type: ignore
            numpy_to_tensor,
        ]),
        'valid': transforms.Compose([
            numpy_to_tensor,
        ]),
        'test': transforms.Compose([
            numpy_to_tensor,
        ])
    }

    return data_transforms

def create_synchronised_audio_data_transforms(fs: int, dataset: str) -> dict[str, transforms.Compose]:

    thickness = 0.04

    if "ticking-heart" not in dataset:
        return None

    data_transforms = {
        'train': transforms.Compose([
            RandomStretchSynchronized(fs),
            RandomPCGAugmentSynchronised(fs),
            RandomTimeFreqMaskSynchronised(thickness, fs), # type: ignore,
            RandomTimeFreqMaskSynchronised(thickness / 2, fs), # type: ignore
            numpy_to_tensor_tuple,
        ]),
        'valid': transforms.Compose([
            numpy_to_tensor_tuple,
        ]),
        'test': transforms.Compose([
            numpy_to_tensor_tuple,
        ])
    }

    return data_transforms
