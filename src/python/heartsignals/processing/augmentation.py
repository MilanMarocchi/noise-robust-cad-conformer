"""
    augmentation.py
    Author: Leigh Abbott

    Purpose: Data augmentation on pcg and ecg signals
"""
import copy
import logging
from typing import Optional

import torchaudio
from processing.filtering import (
    minmax_normalise_signal,
    resample,
    standardise_signal,
    stretch_resample,
    random_parametric_eq,
    random_crop,
    time_stretch_crop,
)
from util.paths import (
    EPHNOGRAM,
    MIT,
)
from transformers import Wav2Vec2Model
from transformers import logging as hf_logging
from scipy.interpolate import CubicSpline
import torch
import torch.optim as optim
import torch.nn as nn
import librosa
import random
import numpy as np
import scipy.signal as ssg
import wfdb
import glob
import os

hf_logging.set_verbosity_error()

def randfloat(low: float, high: float) -> float:
    return low + random.random() * (high - low)


def get_record(path: str, max_sig_len_s: float = -1.0) -> wfdb.Record:

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_sig_len_s <= -1.0:
        target_sig_len = sig_len
    else:
        target_sig_len = round(max_sig_len_s * fs) # type: ignore

    if sig_len > target_sig_len:
        sampfrom = random.randint(0, sig_len - target_sig_len)
        sampto = sampfrom + target_sig_len
    else:
        sampfrom = 0
        sampto = sig_len

    rec = wfdb.rdrecord(path, sampfrom=sampfrom, sampto=sampto)
    return rec


def get_pcg_noise(target_sr: float, len_record: int, path: str = "", reduce_noise: bool = False) -> np.ndarray:

    if path == "":
        path = EPHNOGRAM
    valid_files = glob.glob(f"{path}/*.hea")

    num_tries = 0

    while num_tries < 50:

        try:

            num_tries += 1

            valid_file = random.choice(valid_files)

            record = get_record(valid_file.removesuffix('.hea'))
            pcg_noise_1 = record.p_signal[:, record.sig_name.index('AUX1')] # type: ignore
            pcg_noise_2 = record.p_signal[:, record.sig_name.index('AUX2')] # type: ignore
            pcg_noise_1 = ssg.resample_poly(pcg_noise_1, target_sr, record.fs)
            pcg_noise_2 = ssg.resample_poly(pcg_noise_2, target_sr, record.fs)
            pcg_noise_1 = standardise_signal(random_crop(pcg_noise_1, len_record))
            pcg_noise_2 = standardise_signal(random_crop(pcg_noise_2, len_record))
            pcg_noise_1 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_1
            pcg_noise_2 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_2

            pcg_comb_noise = pcg_noise_1 + pcg_noise_2
            # Try to avoid the divide by 0 in standardise_signal
            if np.max(np.abs(pcg_comb_noise)) > 0.0:
                pcg_comb_noise = standardise_signal(pcg_comb_noise)

            # Reduce the noise of the signal
            if reduce_noise:
                pcg_comb_noise *= 0.15

            return pcg_comb_noise

        except ValueError:
            pass

    return np.zeros(len_record)

def get_pcg_signal(len_record: int, fs: int, path: str = "") -> np.ndarray:
    
    if path == "":
        path = EPHNOGRAM
    valid_files = glob.glob(f"{path}/*.hea")

    num_tries = 0
    while num_tries < 50:

        try:

            num_tries += 1

            valid_file = random.choice(valid_files)

            record = get_record(valid_file.removesuffix('.hea'))
            pcg = record.p_signal[:, record.sig_name.index('PCG')] # type: ignore
            fs_p = record.fs
            pcg = resample(pcg, fs_p, fs)

            return pcg

        except ValueError:
            pass

    return np.random.normal(0, 1, len_record)


def get_ecg_noise(target_sr: float, len_record: int, path: str = "") -> np.ndarray:

    if path == "":
        path = MIT

    em_noise = get_record(os.path.join(path,'em'))
    bw_noise = get_record(os.path.join(path,'bw'))
    ma_noise = get_record(os.path.join(path,'ma'))

    em_noise = ssg.resample_poly(em_noise.p_signal[:, 0], target_sr, em_noise.fs) # type: ignore
    bw_noise = ssg.resample_poly(bw_noise.p_signal[:, 0], target_sr, bw_noise.fs) # type: ignore
    ma_noise = ssg.resample_poly(ma_noise.p_signal[:, 0], target_sr, ma_noise.fs) # type: ignore

    em_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(em_noise, len_record))
    bw_noise = random.choice([0, randfloat(0.0, 0.5)]) * standardise_signal(random_crop(bw_noise, len_record))
    ma_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(ma_noise, len_record))

    return em_noise + bw_noise + ma_noise


def augment_style(content_signal: np.ndarray, style_signal: np.ndarray, cnn_extractor, device) -> np.ndarray:

    torch.cuda.empty_cache()

    min_len = min(len(content_signal), len(style_signal))

    content_audio = torch.tensor(content_signal[:min_len], dtype=torch.float32).clone()
    style_audio = torch.tensor(style_signal[:min_len], dtype=torch.float32).clone()

    content_audio = content_audio.unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, sequence_length]
    style_audio = style_audio.unsqueeze(0).unsqueeze(0).to(device)

    # Function to compute Gram matrix
    def gram_matrix(tensor):
        b, c, t = tensor.size()
        features = tensor.view(b * c, t)
        G = torch.mm(features, features.t()) / (b * c * t)
        return G

    # Functions to extract content and style features
    def get_content_features(features, content_layers):
        return {layer: features[layer] for layer in content_layers}

    def get_style_features(features, style_layers):
        style_features = {}
        for layer in style_layers:
            feature = features[layer]
            gram = gram_matrix(feature)
            style_features[layer] = gram
        return style_features

    # Extract features for content and style audio
    with torch.no_grad():
        content_features_all = cnn_extractor(content_audio)
        style_features_all = cnn_extractor(style_audio)

    # Define layers to be used for content and style
    # You can adjust these layers based on experimentation
    content_layers = ['conv_4']
    style_layers = ['conv_0', 'conv_1', 'conv_2', 'conv_3']

    # Get content and style features
    content_features = get_content_features(content_features_all, content_layers)
    style_features = get_style_features(style_features_all, style_layers)

    # Initialize generated audio with random noise
    generated_audio = torch.randn_like(content_audio, requires_grad=True).to(device)

    # Set up optimizer and loss weights
    optimizer = optim.RMSprop([generated_audio], lr=0.01)
    content_weight = 1e2
    style_weight = 1e3

    # Optimization loop
    num_steps = 2000
    for _ in (range(num_steps)):
        optimizer.zero_grad()
        
        # Forward pass through CNN feature extractor
        gen_features = cnn_extractor(generated_audio)
        
        # Compute content loss
        content_loss = 0
        gen_content_features = get_content_features(gen_features, content_layers)
        for layer in content_layers:
            content_loss += nn.MSELoss()(gen_content_features[layer], content_features[layer])
        
        # Compute style loss
        style_loss = 0
        gen_style_features = get_style_features(gen_features, style_layers)
        for layer in style_layers:
            style_loss += nn.MSELoss()(gen_style_features[layer], style_features[layer])
        
        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()
        
        # Clamp generated audio to valid range
        #with torch.no_grad():
        #    generated_audio.clamp_(-6.0, 6.0)
    optimizer.zero_grad()
        
    # Denormalize the generated audio
    generated_audio_data = generated_audio.detach().cpu().numpy().squeeze()
    generated_audio_data = standardise_signal(generated_audio_data)

    # cleanup
    del generated_audio, optimizer, content_audio, style_audio, content_features, style_features, gen_features, gen_style_features, gen_content_features, content_features_all, style_features_all
    torch.cuda.empty_cache()

    return generated_audio_data

def augment_multi_pcg_style(pcg: list[np.ndarray], fs: int, device: str) -> list[np.ndarray]:
    pcg_multi_wav = copy.deepcopy(pcg)

    device = device if torch.cuda.is_available() else "cpu"
    # Load pre-trained wav2vec2 model
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.gradient_checkpointing_disable()
    model.eval()

    # Access the CNN feature extractor
    feature_extractor = model.feature_extractor

    # Define a helper function to get intermediate CNN layer outputs
    class CNNFeatureExtractor(nn.Module):
        def __init__(self, cnn):
            super(CNNFeatureExtractor, self).__init__()
            self.layers = nn.ModuleList(cnn.conv_layers)
            self.layer_names = [f'conv_{i}' for i in range(len(self.layers))]

        def forward(self, x):
            features = {}
            for i, layer in enumerate(self.layers):
                x = layer(x)
                features[self.layer_names[i]] = x
            return features

    cnn_extractor = CNNFeatureExtractor(feature_extractor).to(device)
    cnn_extractor.eval()

    for param in cnn_extractor.parameters():
        param.requires_grad = False
        param.grad = None

    style_signal = get_pcg_signal(len(pcg_multi_wav[0]), fs)
    torch.cuda.empty_cache() # Type to stop memory leaks
    for idx, pcg_wav in enumerate(pcg_multi_wav):
        try:
            pcg_multi_wav[idx] = standardise_signal(augment_style(pcg_wav, style_signal, cnn_extractor, device))
        except Exception:
            torch.cuda.empty_cache()
            raise Exception("Issue with GPU allocation.")
        torch.cuda.empty_cache() # Type to stop memory leaks

    del model, cnn_extractor, feature_extractor
    torch.cuda.empty_cache() # Type to stop memory leaks

    return pcg_multi_wav

def augment_hpss(pcg_wav: np.ndarray,
                 n_fft_1: Optional[int] = None, win_len_1: Optional[int] = None,
                 hop_len_1: Optional[int] = None, margin_1: Optional[int] = None,
                 kernel_1: Optional[int] = None, n_fft_2: Optional[int] = None,
                 win_len_2: Optional[int] = None, hop_len_2: Optional[int] = None, margin_2: Optional[int] = None,
                 kernel_2: Optional[int] = None
) -> tuple[np.ndarray, int]:
    n_fft_1 = random.choice([512, 1024, 2048]) if n_fft_1 is None else n_fft_1
    win_len_1 = n_fft_1
    hop_len_1 = random.choice([16, 32, 64, 128]) if hop_len_1 is None else hop_len_1
    margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0)) if margin_1 is None else margin_1
    kernel_1 = (random.randint(5, 30), random.randint(5, 30)) if kernel_1 is None else kernel_1

    n_fft_2 = random.choice([512, 1024, 2048]) if n_fft_2 is None else n_fft_2
    win_len_2 = n_fft_2
    hop_len_2 = random.choice([16, 32, 64, 128]) if hop_len_2 is None else hop_len_2
    margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0)) if margin_2 is None else margin_2
    kernel_2 = (random.randint(5, 30), random.randint(5, 30)) if kernel_2 is None else kernel_2

    decomp = librosa.stft(
        pcg_wav,
        n_fft=n_fft_1,
        hop_length=hop_len_1,
        win_length=win_len_1,
    )

    harmon, percus = librosa.decompose.hpss(
        decomp,
        margin=margin_1,
        kernel_size=kernel_1,
    )
    resid = decomp - (harmon + percus)

    y_1 = librosa.istft(
        harmon,
        n_fft=n_fft_1,
        hop_length=hop_len_1,
        win_length=win_len_1,
    )

    y_2 = librosa.istft(
        percus,
        n_fft=n_fft_1,
        hop_length=hop_len_1,
        win_length=win_len_1,
    )

    y_3 = librosa.istft(
        resid,
        n_fft=n_fft_1,
        hop_length=hop_len_1,
        win_length=win_len_1,
    )

    decomp = librosa.stft(
        y_1,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    harmon, percus = librosa.decompose.hpss(
        decomp,
        margin=margin_2,
        kernel_size=kernel_2,
    )
    resid = decomp - (harmon + percus)

    y_11 = librosa.istft(
        harmon,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    y_12 = librosa.istft(
        percus,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    y_13 = librosa.istft(
        resid,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    decomp = librosa.stft(
        y_2,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    harmon, percus = librosa.decompose.hpss(
        decomp,
        margin=margin_2,
        kernel_size=kernel_2,
    )
    resid = decomp - (harmon + percus)

    y_21 = librosa.istft(
        harmon,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    y_22 = librosa.istft(
        percus,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    y_23 = librosa.istft(
        resid,
        n_fft=n_fft_2,
        hop_length=hop_len_2,
        win_length=win_len_2,
    )

    min_len = min(len(y_i) for y_i in (y_11, y_12, y_13, y_21, y_22, y_23, y_3))

    pcg_wav_1 = standardise_signal(
        1 * randfloat(0.01, 10)*y_11[:min_len]
        + 1 * randfloat(0.01, 10)*y_12[:min_len]
        + 1 * randfloat(0.01, 10)*y_13[:min_len]
        + 1 * randfloat(0.01, 10)*y_21[:min_len]
        + 1 * randfloat(0.01, 10)*y_22[:min_len]
        + 1 * randfloat(0.01, 10)*y_23[:min_len]
        + 1 * randfloat(0.01, 10)*y_3[:min_len]
    )

    pcg_wav_2 = standardise_signal(
        1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_13[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_23[:min_len])
        + 1 * randfloat(0.01, 10)*standardise_signal(y_3[:min_len])
    )

    pcg_wav = standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2)

    return pcg_wav, min_len


def augment_rand_noise(signal: np.ndarray) -> np.ndarray:
    noise_std = random.choice([0.0001, 0.001, 0.01])
    signal += randfloat(0, 0.1) * np.random.normal(0, noise_std, signal.shape)
    signal = standardise_signal(signal)

    return signal 

def augment_amplitude_warp(signal: np.ndarray, num_control_points: int = 12, 
                           amplitude_range: tuple = (0.7, 1.3), random_amplitudes: Optional[np.ndarray] = None):
    """
    Apply random magnitude warping to a signal.
    """
    # NOTE: This is very similar to the sin_envelope one.
    N = len(signal)
    # Generate evenly spaced control points
    control_points = np.linspace(0, N - 1, num_control_points)
    # Generate random amplitudes for control points
    if random_amplitudes is None:
        random_amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], size=num_control_points)
    # Create a cubic spline interpolation of the random amplitudes
    cs = CubicSpline(control_points, random_amplitudes, bc_type='natural')
    # Generate the amplitude modulation curve
    amplitude_curve = cs(np.arange(N))
    # Apply the amplitude modulation to the signal
    spline_curve = amplitude_curve /  np.sum(amplitude_curve)
    #warped_signal = signal * amplitude_curve
    warped_signal = np.convolve(signal, spline_curve, mode='same')
    return warped_signal

def augment_time_warp(signal: np.ndarray, sr: int, min_factor: float = 0.8, max_factor: float = 1.2, time_stretch_factor: Optional[float] = None) -> np.ndarray:
    if time_stretch_factor is None:
        time_stretch_factor = randfloat(min_factor, max_factor)
    signal = stretch_resample(signal, sr, time_stretch_factor)
    signal = standardise_signal(signal)
    return signal


def augment_banding(signal: np.ndarray, sr: int, f_low: int, f_high: 500) -> np.ndarray:
    signal = random_parametric_eq(signal, sr, low=f_low, high=f_high)
    signal = standardise_signal(signal)
    return signal


def augment_sin_envelope(signal: np.ndarray, sr: int, a_low: float = 0.01, a_high: float = 0.25) -> np.ndarray:
    t = np.arange(signal.size) / sr
    vol_mod_1 = randfloat(a_low, a_high) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
    vol_mod_2 = randfloat(a_low, a_high) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
    signal *= (1 + vol_mod_1 + vol_mod_2)
    signal = standardise_signal(signal)
    return signal

def augment_baseline_wander(signal: np.ndarray, sr: int) -> np.ndarray:
    t = np.arange(signal.size) / sr
    baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
    baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
    signal += baseline_wander
    signal = standardise_signal(signal)
    return signal

def augment_pcg(orig_pcg_wav: np.ndarray, sr: int,
                    prob_noise: float = 0.30, 
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.45,
                    prob_hpss: float = 0.75, prob_mag_warp: float = 0.45,
                    prob_banding: float = 0.25, prob_real_noise: float = 0.35,
                    time_stretch_min: float = 0.8, time_stretch_max: float = 1.2,
                    EPHNOGRAM="") -> np.ndarray:

    pcg_wav = orig_pcg_wav.copy()
    pcg_wav = minmax_normalise_signal(pcg_wav)

    if np.random.rand() < prob_hpss:
        pcg_wav, _ = augment_hpss(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        pcg_wav = augment_rand_noise(pcg_wav)

    if np.random.rand() < prob_time_warp:
        pcg_wav = augment_time_warp(pcg_wav, sr, time_stretch_min, time_stretch_max)

    if np.random.rand() < prob_wandering_volume:
        pcg_wav = augment_sin_envelope(pcg_wav, sr, 0.01, 0.25)

    if np.random.rand() < prob_noise / 4:
        pcg_wav = augment_rand_noise(pcg_wav)

    if np.random.rand() < prob_mag_warp:
        pcg_wav = augment_amplitude_warp(pcg_wav)

    if np.random.rand() < prob_banding:
        pcg_wav = augment_banding(pcg_wav, sr, f_low=2, f_high=500)

    if np.random.rand() < prob_real_noise:
        pcg_wav += get_pcg_noise(sr, len(pcg_wav), EPHNOGRAM)

    return pcg_wav


def augment_ecg(orig_ecg_wav: np.ndarray, sr: int,
                prob_noise: float = 0.3, prob_baseline_wander: float = 0.3,
                prob_time_warp: float = 0.25, prob_banding: float = 0.25,
                prob_real_noise: float = 0.5,
                MIT=""
) -> np.ndarray:

    ecg_wav = orig_ecg_wav.copy()
    ecg_wav = minmax_normalise_signal(ecg_wav)

    if np.random.rand() < prob_noise / 4:
        ecg_wav = augment_rand_noise(ecg_wav)

    if np.random.rand() < prob_baseline_wander:
        t = np.arange(ecg_wav.size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        ecg_wav += baseline_wander
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.8, 1.2)
        ecg_wav = augment_time_warp(ecg_wav, sr, time_stretch_factor=time_stretch_factor)

    if np.random.rand() < prob_noise / 4:
        ecg_wav = augment_rand_noise(ecg_wav)

    if np.random.rand() < prob_banding:
        ecg_wav = augment_banding(ecg_wav, sr, f_low=0.25, f_high=100)

    if np.random.rand() < prob_real_noise:
        ecg_wav += get_ecg_noise(sr, len(ecg_wav), MIT)

    return standardise_signal(ecg_wav)

def augment_signals(orig_ecg_wav: np.ndarray, orig_pcg_wav: np.ndarray, sr: int,
                    prob_noise: float = 0.30, prob_baseline_wander: float = 0.30,
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.25,
                    prob_hpss: float = 0.75, prob_banding: float = 0.25,
                    prob_real_noise: float = 0.5,
                    MIT="", EPHNOGRAM=""
) -> tuple[np.ndarray, np.ndarray]:

    ecg_wav = orig_ecg_wav.copy()
    pcg_wav = orig_pcg_wav.copy()

    ecg_wav = minmax_normalise_signal(ecg_wav)
    pcg_wav = minmax_normalise_signal(pcg_wav)

    if np.random.rand() < prob_hpss:
        pcg_wav, min_len = augment_hpss(pcg_wav)
        ecg_wav = ecg_wav[:min_len]

    if np.random.rand() < prob_noise / 4:
        pcg_wav = augment_rand_noise(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        ecg_wav = augment_rand_noise(ecg_wav)

    if np.random.rand() < prob_baseline_wander:
        t = np.arange(ecg_wav.size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        ecg_wav += baseline_wander
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.8, 1.2)
        ecg_wav = augment_time_warp(ecg_wav, sr, time_stretch_factor=time_stretch_factor)
        pcg_wav = augment_time_warp(pcg_wav, sr, time_stretch_factor=time_stretch_factor)

    if np.random.rand() < prob_wandering_volume:
        pcg_wav = augment_sin_envelope(pcg_wav, sr)

    if np.random.rand() < prob_noise / 4:
        pcg_wav = augment_rand_noise(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        ecg_wav = augment_rand_noise(ecg_wav)

    if np.random.rand() < prob_banding:
        pcg_wav = augment_banding(pcg_wav, sr, f_low=2, f_high=500)

    if np.random.rand() < prob_banding:
        ecg_wav = augment_banding(ecg_wav, sr, f_low=0.25, f_high=100)

    if np.random.rand() < prob_real_noise:
        ecg_wav += get_ecg_noise(sr, len(ecg_wav), MIT)

    if np.random.rand() < prob_real_noise:
        pcg_wav += get_pcg_noise(sr, len(pcg_wav), EPHNOGRAM)

    return ecg_wav, pcg_wav

def augment_synchronised_pcg(orig_pcg1: np.ndarray, orig_pcg2: np.ndarray, sr: int,
                    prob_noise: float = 0.30, 
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.75,
                    prob_hpss: float = 0.75, prob_mag_warp: float = 0.5,
                    prob_banding: float = 0.25, prob_real_noise: float = 0.35,
                    EPHNOGRAM="") -> tuple:
    """
    For multichannel pcg recordings, to ensure the same augmentation occurs on all channels
    """
    assert not np.all(orig_pcg1 == 0), "Signal pcg1 is all 0"
    assert not np.all(orig_pcg2 == 0), "Signal pcg2 is all 0"

    pcg_wav1 = orig_pcg1.copy()
    pcg_wav2 = orig_pcg2.copy()

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])

        pcg_wav1 += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav1.shape)
        pcg_wav2 += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav2.shape)
        pcg_wav1 = standardise_signal(pcg_wav1)
        pcg_wav2 = standardise_signal(pcg_wav2)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_wav1.size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        pcg_wav1 *= (1 + vol_mod_1 + vol_mod_2)
        pcg_wav2 *= (1 + vol_mod_1 + vol_mod_2)
        pcg_wav1 = standardise_signal(pcg_wav1)
        pcg_wav2 = standardise_signal(pcg_wav2)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        coefficient = randfloat(0, 0.1)

        pcg_wav1 += coefficient * np.random.normal(0, noise_std, pcg_wav1.shape)
        pcg_wav2 += coefficient * np.random.normal(0, noise_std, pcg_wav2.shape)
        pcg_wav1 = standardise_signal(pcg_wav1)
        pcg_wav2 = standardise_signal(pcg_wav2)

    if np.random.rand() < prob_banding:
        pcg_wav1 = random_parametric_eq(pcg_wav1, sr, low=2, high=500)
        pcg_wav2 = random_parametric_eq(pcg_wav2, sr, low=2, high=500)
        pcg_wav1 = standardise_signal(pcg_wav1)
        pcg_wav2 = standardise_signal(pcg_wav2)

    if np.random.rand() < prob_real_noise:
        pcg_noise = get_pcg_noise(sr, len(pcg_wav1), EPHNOGRAM)
        pcg_wav1 += pcg_noise
        pcg_wav2 += pcg_noise
        pcg_wav1 = standardise_signal(pcg_wav1)
        pcg_wav2 = standardise_signal(pcg_wav2)

    return pcg_wav1, pcg_wav2


def augment_multi_pcg(orig_multi_pcg_wav: list, sr: int,
                    prob_noise: float = 0.30, 
                    prob_wandering_volume: float = 0.75, prob_time_warp: float = 0.35,
                    prob_hpss: float = 0.75, prob_mag_warp: float = 0.5,
                    prob_banding: float = 0.25, prob_real_noise: float = 0.25,
                    EPHNOGRAM="") -> list[np.ndarray]:
    """
    For multichannel pcg recordings, to ensure the same augmentation occurs on all channels
    """
    for pcg in orig_multi_pcg_wav:
        assert not np.all(pcg == 0), "Signal pcg is all 0"

    pcg_multi_wav: list[np.ndarray] = list()

    for orig_pcg_wav in orig_multi_pcg_wav:
        pcg_wav = orig_pcg_wav.copy()
        pcg_wav = standardise_signal(pcg_wav)
        pcg_multi_wav.append(pcg_wav)

    #if np.random.rand() < prob_hpss:

    #    n_fft_1 = random.choice([512, 1024, 2048])
    #    win_len_1 = n_fft_1
    #    hop_len_1 = random.choice([16, 32, 64, 128])
    #    margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
    #    kernel_1 = (random.randint(5, 30), random.randint(5, 30))

    #    n_fft_2 = random.choice([512, 1024, 2048])
    #    win_len_2 = n_fft_2
    #    hop_len_2 = random.choice([16, 32, 64, 128])
    #    margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
    #    kernel_2 = (random.randint(5, 30), random.randint(5, 30))

    #    for idx, pcg_wav in enumerate(pcg_multi_wav):

    #        augmented_pcg_wav, _ = augment_hpss(
    #            pcg_wav,
    #            n_fft_1=n_fft_1,
    #            win_len_1=win_len_1,
    #            hop_len_1=hop_len_1,
    #            margin_1=margin_1,
    #            kernel_1=kernel_1,
    #            n_fft_2=n_fft_2,
    #            win_len_2=win_len_2,
    #            hop_len_2=hop_len_2,
    #            margin_2=margin_2,
    #            kernel_2=kernel_2
    #        )
    #        pcg_multi_wav[idx] = augmented_pcg_wav 

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(0.7, 1.3)

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = time_stretch_crop(pcg_wav, sr, time_stretch_factor)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    #if np.random.rand() < prob_mag_warp:
    #    num_control_points = 8
    #    random_amplitudes = np.random.uniform(0.85, 1.15, size=num_control_points)
    #    for idx, pcg_wav in enumerate(pcg_multi_wav):
    #        pcg_multi_wav[idx] = standardise_signal(augment_amplitude_warp(pcg_wav, random_amplitudes=random_amplitudes, num_control_points=num_control_points))



    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_multi_wav[0].size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_multi_wav[idx] *= (1 + vol_mod_1 + vol_mod_2)
            pcg_multi_wav[idx] = standardise_signal(pcg_multi_wav[idx])

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_real_noise:
        pcg_noise = get_pcg_noise(sr, len(pcg_multi_wav[0]), EPHNOGRAM)
        for idx in range(len(pcg_multi_wav)):
            pcg_multi_wav[idx] += pcg_noise

    for pcg in pcg_multi_wav:
        assert not np.all(pcg == 0), "Signal pcg is all 0"

    return pcg_multi_wav

    if np.random.rand() < prob_banding:
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)


    return pcg_multi_wav 

def augment_multi_ecg(orig_multi_ecg_wav: list[np.ndarray], sr: int,
                    prob_noise: float = 0.30, prob_baseline_wander: float = 0.30,
                    prob_time_warp: float = 0.25,
                    prob_banding: float = 0.25,
                    prob_real_noise: float = 0.5,
                    time_stretch_factor: Optional[float] = None,
                    MIT=""
) -> list[np.ndarray]:

    ecg_multi_wav = list()

    for orig_pcg_wav in orig_multi_ecg_wav:
        ecg_wav = orig_pcg_wav.copy()
        ecg_wav = standardise_signal(ecg_wav)
        ecg_multi_wav.append(ecg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for ecg_wav in ecg_multi_wav:
            ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
            ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_baseline_wander:
        t = np.arange(ecg_wav.size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for ecg_wav in ecg_multi_wav:
            ecg_wav += baseline_wander
            ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_time_warp:
        if time_stretch_factor is None:
            time_stretch_factor = randfloat(0.94, 1.006)
        for ecg_wav in ecg_multi_wav:
            ecg_wav = stretch_resample(ecg_wav, sr, time_stretch_factor)
            ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for ecg_wav in ecg_multi_wav:
            ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
            ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_banding:
        for ecg_wav in ecg_multi_wav:
            ecg_wav = random_parametric_eq(ecg_wav, sr, low=0.25, high=100)
            ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_real_noise:
        for ecg_wav in ecg_multi_wav:
            ecg_wav += get_ecg_noise(sr, len(ecg_wav), MIT)

    return ecg_multi_wav
