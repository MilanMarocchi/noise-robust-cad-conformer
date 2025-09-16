"""
    preprocessing.py
    Author: Milan Marocchi

    Purpose: Run any preprocessing that is required for signals before training
"""

import torch
from processing.filtering import (
    bandpass,
    interpolate_nans,
    notchfilter,
    resample,
    low_pass_butter,
    high_pass_butter,
    normalise_signal,
    spike_removal_p,
    create_band_filters,
    start_matlab,
    stop_matlab,
    wavelet_denoise,
    wdenoise,
    noise_canc,
)

import logging

import numpy as np
import scipy.signal as ssg
import logging

from util.paths import MATLAB_PATH

def process_pcg_denoise(pcg: np.ndarray, fs: int) -> np.ndarray:
    pcg = wavelet_denoise(pcg, 'db10', 7)
    return pcg


def process_ecg_denoise(ecg: np.ndarray, fs: int) -> np.ndarray:
    ecg = wavelet_denoise(ecg, 'sym4', 4)
    return ecg


def process_noise_cancel(pcg: np.ndarray, pcg_nm: np.ndarray, fs: int) -> np.ndarray:
    pcg_len = len(pcg)
    assert len(pcg) == len(pcg_nm), "Mismatch in NM and HM lengths"
    pcg = noise_canc(pcg_nm, pcg, fs=fs, FL=256) 

    return pcg


def pre_process_pcg(pcg: np.ndarray, fs: int, fs_new: int) -> np.ndarray:
    pcg = interpolate_nans(pcg)
    pcg = resample(pcg, fs, fs_new) if fs != fs_new else pcg
    pcg = spike_removal_p(pcg, fs_new)

    pcg = low_pass_butter(pcg, 2, 450, fs_new)
    pcg = high_pass_butter(pcg, 2, 25, fs_new)
    pcg = normalise_signal(pcg)
    # FIXME: Get this working
    #pcg = wavelet_denoise(pcg, 'db10', 7)

    return pcg


def pre_process_ecg(ecg: np.ndarray, fs: int, fs_new: int) -> np.ndarray:
    ecg = interpolate_nans(ecg)
    ecg = resample(ecg, fs, fs_new)

    ecg = low_pass_butter(ecg, 2, 40, fs_new)
    ecg = high_pass_butter(ecg, 2, 2, fs_new)
    ecg = normalise_signal(ecg)

    #ecg = pre_filter_ecg(ecg, fs_new)
    # FIXME: Get this working
    #ecg = wavelet_denoise(ecg, 'sym4', 4)

    return ecg


def pre_process_orig_four_bands(pcg: np.ndarray, fs: int) -> np.ndarray:
    data = np.zeros((len(pcg), 4))

    pcg = pcg.squeeze()

    b = create_band_filters(fs)
    for i in range(4):
        data[:, i] = ssg.filtfilt(b[i], 1, pcg)

    return data


def pre_filter_ecg(signal, fs):
    signal = notchfilter(signal, fs, 50, 55)
    signal = notchfilter(signal, fs, 60, 55)
    signal = notchfilter(signal, fs, 100, 55)
    signal = notchfilter(signal, fs, 120, 55)
    signal = bandpass(signal, fs, 0.25, 150)
    # signal = wavefilt(signal, 'sym4', 4)
    # signal = bandpass(signal, fs, 0.5, 70)
    return signal


def mid_filter_ecg(signal, fs):
    signal = bandpass(signal, fs, 0.25, 150)
    return signal


def post_filter_ecg(signal, fs):
    signal = notchfilter(signal, fs, 50, 55)
    signal = notchfilter(signal, fs, 60, 55)
    signal = notchfilter(signal, fs, 100, 55)
    signal = notchfilter(signal, fs, 120, 55)
    return bandpass(signal, fs, 0.25, 70)


def pre_filter_pcg(signal, fs):
    signal = bandpass(signal, fs, 2, 500)
    # signal = spike_removal(signal, fs)
    # signal = wavefilt(signal, 'db10', 4)
    # signal = bandpass(signal, fs, 5, 400)
    return signal


def mid_filter_pcg(signal, fs):
    signal = bandpass(signal, fs, 2, 500)
    return signal


def post_filter_pcg(signal, fs):
    return bandpass(signal, fs, 5, 450)


def pcg_filter_classify(signal, fs):
    signal = bandpass(signal, fs, 25, 450)

    return signal


def normalise_array_length(array, normalised_length):
    """
    Pad or crop the first dimension of the array or tensor to normalised_length.
    
    :param array: The input NumPy array or PyTorch tensor.
    :param normalised_length: The target length for the first dimension.
    :return: Tuple of (padded/cropped array or tensor, index before padding).
    """
    is_tensor = torch.is_tensor(array)
    orig_length = array.shape[0]
    pad_amount = max(0, normalised_length - orig_length)

    if pad_amount > 0:
        # Padding
        if is_tensor:
            pad_shape = (pad_amount, *array.shape[1:])
            padding = torch.zeros(pad_shape, dtype=array.dtype, device=array.device)
            array = torch.cat((array, padding), dim=0)
        else:
            pad_shape = ((0, pad_amount),) + tuple((0, 0) for _ in range(array.ndim - 1))
            array = np.pad(array, pad_shape, mode='constant')
    elif orig_length > normalised_length:
        # Cropping
        array = array[:normalised_length]

    pad_idx = min(orig_length, normalised_length)
    return array, pad_idx


def normalise_2d_array_length(array, normalised_length):
    """
    Pad or crop the array to have a shape of (2500, second_dim_size).

    :param array: The input array.
    :param normalised_length: Length to normalise array to.
    :return: Array with shape (2500, second_dim_size).
    """
    pad_amount = 0
    # Pad or crop the first dimension to 2500
    if array.shape[0] < normalised_length:
        # Pad
        pad_amount = normalised_length - array.shape[0]
        array = np.pad(array, ((0, pad_amount), (0, 0)), mode='constant')
    elif array.shape[0] > normalised_length:
        # Crop
        array = array[:normalised_length, :]

    pad_idx = len(array) - pad_amount

    return array, pad_idx
