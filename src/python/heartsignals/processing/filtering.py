"""
    filtering.py
    Author : Milan Marocchi

    Various filters
"""

import copy
from typing import Any, Optional
import numpy as np
import os
import math
import scipy
import scipy.signal as ssg
from scipy.io import loadmat 
import random
import librosa
import pywt
import pyrubberband as pyrb
import torch

import logging
logging.basicConfig(level=logging.INFO)

# Matlab engine
ENG = None

def start_matlab(matlab_location: str):
    print(matlab_location)
    if matlab_location != '':
        try:
            import matlab.engine
            global ENG
            ENG = matlab.engine.start_matlab()
            ENG.addpath(ENG.genpath(str(matlab_location)), nargout=0)  # type: ignore
            logging.info('STARTED MATLAB')
        except ImportError as e:
            logging.error('Matlab engine not installed --- trying anyway')
            logging.error(e)


def stop_matlab():
    if ENG is not None:
        ENG.exit()  # type: ignore
        logging.info('STOPPED MATLAB')


def stretch_resample(signal: np.ndarray, sample_rate: int, time_stretch_factor: float) -> np.ndarray:
    signal = pyrb.time_stretch(signal, sample_rate, rate=time_stretch_factor)
    return signal

def time_stretch_crop(signal: np.ndarray, fs: int, time_stretch_factor: float) -> np.ndarray:
    """Time stretches the signal and crops it to it's original length"""
    sig_len = len(signal)
    signal = pyrb.time_stretch(signal, fs, time_stretch_factor)
    
    return signal[:sig_len]


def random_crop(signal: np.ndarray, len_crop: int) -> np.ndarray:
    start = random.randint(0, len(signal) - len_crop)
    end = start + len_crop
    return signal[start:end]


def random_parametric_eq(signal: np.ndarray, sr: float, low: float, high: float, num_bands: int = 5) -> np.ndarray:
    equalised_signal = np.copy(signal)

    for _ in range(num_bands):

        b_low = np.random.uniform(low=low, high=0.95*high)
        b_high = random.choice([np.random.uniform(low=b_low+0.05*(high-low), high=high), b_low+(high-low)/num_bands])

        sos = ssg.iirfilter(N=1, Wn=[b_low / (sr / 2), b_high / (sr / 2)], btype='band',
                            analog=False, ftype='butter', output='sos')

        equalised_signal = np.asarray(ssg.sosfilt(sos, equalised_signal))

    return standardise_signal(standardise_signal(equalised_signal)/50 + standardise_signal(signal))


def band_stop(signal: np.ndarray, fs: int, fs_low: int, fs_high: int, order:int = 4):
   b, a = ssg.butter(order, [fs_low / fs, fs_high / fs], btype='bandstop') 
   signal = ssg.filtfilt(b, a, signal)

   return signal


def interpolate_nans(a: np.ndarray) -> np.ndarray:
    mask = np.isnan(a)
    a[mask] = np.interp(np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        a[~mask])
    return a


def minmax_normalise_signal(signal: np.ndarray, new_min : float = -1.0, new_max : float = 1.0) -> np.ndarray:
    """min-max normalisation"""
    signal = (signal - signal.min()) / (signal.max() - signal.min()) * (new_max - new_min) + new_min

    return signal


def znormalise_signal(signal: np.ndarray) -> np.ndarray:
    """Normalisation based on z-score"""
    means = signal.mean(axis=0) # type: ignore
    stds = signal.std(axis=0) # type: ignore
    signal = (signal - means) / stds

    return signal


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Naive abs-max normalisation"""
    signal = interpolate_nans(signal)
    signal -= np.mean(signal)
    # To avoid divid by 0
    if np.max(np.abs(signal)) > 0:
        signal /= np.max(np.abs(signal))
    signal = np.clip(signal, -1, 1)

    return signal

def kpeak_normalise_signal_torch(signal: torch.Tensor, k: int = 26, a: float = -1.0, b: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Informed normalisation based on the mean of k-peaks using PyTorch and topk"""

    # Get k largest values
    k_largest = torch.topk(signal, k=k, dim=dim, largest=True).values
    # Get k smallest values by negating and taking topk
    k_smallest = torch.topk(-signal, k=k, dim=dim, largest=True).values
    k_smallest = -k_smallest

    k_min = k_smallest.mean()
    k_max = k_largest.mean()

    # Apply the normalization
    normalized_signal = a + ((signal - k_min) / (k_max - k_min)) * (b - a)

    return normalized_signal

def kpeak_normalise_signal(signal: np.ndarray, k: int = 3, a: int = -1, b: int = 1) -> np.ndarray:
    """Informed normalisation based on the mean of k-peaks"""
    #signal = interpolate_nans(signal)
    
    sorted = np.sort(signal)
    k_smallest = sorted[:k]
    k_largest = sorted[-k:]

    k_min = np.average(k_smallest)
    k_max = np.average(k_largest)

    # From Enhancing cross-domain robustness in phonocardiogram signal classification using domain-invariant preprocessing and transfer learning
    signal = a + ((signal - k_min) * (k_max - k_min)) * (b - a)

    return signal

def minmax_normalise_signal(signal: np.ndarray, new_min : float = -1.0, new_max : float = 1.0) -> np.ndarray:
    """min-max normalisation"""
    signal = (signal - signal.min()) / ((signal.max() - signal.min()) * (new_max - new_min) + new_min + 1e-8)

    return signal


def standardise_signal(signal: np.ndarray) -> np.ndarray:
    return normalise_signal(signal)


def standardise_torch_signal(signal):
    return torch.from_numpy(standardise_signal(signal.cpu().numpy())).squeeze(0).float()


def bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low /= nyquist_freq
    high /= nyquist_freq

    sos = ssg.butter(1, [low, high], 'bandpass', analog=False, output='sos',)
    signal = ssg.sosfiltfilt(sos, signal)

    return signal


def notchfilter(signal: np.ndarray, fs: float, notch: float, Q: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    notch /= nyquist_freq

    b, a = ssg.iirnotch(notch, Q)
    signal = ssg.filtfilt(b, a, signal)

    return signal


def get_pcg_segs_idx(pcg, old_fs, new_fs):
    pcg = resample(pcg, old_fs, new_fs)

    pcg = ENG.butterworth_low_pass_filter(pcg, 2, 400, new_fs)  # type: ignore
    pcg = ENG.butterworth_high_pass_filter(pcg, 2, 25, new_fs)  # type: ignore
    pcg = np.array(pcg).reshape(-1, 1)
    pcg = ENG.schmidt_spike_removal(pcg, float(new_fs))  # type: ignore

    assigned_states = ENG.segmentation(pcg, new_fs)  # type: ignore
    seg_idxs = np.asarray(ENG.get_states(assigned_states), dtype=int) - 1  # type: ignore

    return seg_idxs, new_fs


def create_band_filters(fs: int) -> list[np.ndarray]:
    N = 61
    sr = fs
    wn = 45 * 2 / sr
    b1 = ssg.firwin(N, wn, window='hamming', pass_zero='lowpass') # type: ignore
    wn = [45 * 2 / sr, 80 * 2 / sr]
    b2 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = [80 * 2 / sr, 200 * 2 / sr]
    b3 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = 200 * 2 / sr
    b4 = ssg.firwin(N, wn, window='hamming', pass_zero='highpass') # type: ignore

    return [b1, b2, b3, b4]


def hilbert_envelope(signal: np.ndarray, fs: int) -> np.ndarray:
    envelope = copy.deepcopy(signal)
    envelope = np.abs(ssg.hilbert(envelope))

    return envelope


def homomorphic_envelope(signal: np.ndarray, fs: int, order: int = 6, cutoff: float = 8) -> np.ndarray:
    """
    Note the order and cutoff are for a lowpass filter
    """
    if cutoff > fs / 2:
        raise ValueError(f"Invalid cutoff {cutoff=} for the sampling frequency {fs=}")

    env = hilbert_envelope(signal, fs)

    low_pass = ssg.butter(order, cutoff, output='sos', fs=fs, btype='lowpass')
    env[env<=0] = np.finfo(float).eps

    filtered_env = ssg.sosfiltfilt(low_pass, np.log(env))

    homomorphic_env = np.exp(filtered_env)

    return homomorphic_env


def spike_removal_p(original_signal: np.ndarray, fs: float) -> np.ndarray:
    """Python implementation of schmidt spike removal"""
    initial_shape = original_signal.shape

    # Find the window size (500 ms)
    windowsize = round(float(fs) / 2.0)

    # Find any samples outside of an integer number of windows
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows
    sampleframes = original_signal[:len(original_signal) - trailingsamples].reshape((windowsize, -1), order='F')

    # Find the Maximum Absolute Amplitudes (MAAs)
    MAAs = np.max(np.abs(sampleframes), axis=0)

    max_iterations = 1000  
    iteration = 0

    while np.any(MAAs > np.median(MAAs) * 3) and iteration < max_iterations:
        previous_MAAs = MAAs.copy()
        window_num = np.argmax(MAAs)
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        zero_crossings = np.concatenate([(np.abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1), np.zeros(1)])

        spike_start = 0 if not np.any(zero_crossings[:spike_position]) else np.where(zero_crossings[:spike_position])[0][-1] + 1
        zero_crossings[:spike_position] = 0
        spike_end = windowsize - 1 if not np.any(zero_crossings) else np.where(zero_crossings)[0][0]

        sampleframes[spike_start:spike_end, window_num] = 0.0001

        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

        # Stop if no change in MAAs (convergence)
        if np.array_equal(MAAs, previous_MAAs):
            break
        
        iteration += 1

    # Reshape the despiked signal back to its original form
    despiked_signal = sampleframes.flatten(order='F')

    # Add trailing samples back
    despiked_signal = np.concatenate((despiked_signal, original_signal[len(despiked_signal):]))

    assert despiked_signal.shape == initial_shape, "Shape should remain unchanged" 

    return despiked_signal


def spike_removal(signal: np.ndarray, fs: float, matlab_location: str = "") -> np.ndarray:
    signal = np.array(signal).reshape(-1, 1)
    signal = ENG.schmidt_spike_removal(signal, float(fs))  # type: ignore
    signal = np.asarray(signal).flatten()
    return signal


def fade_signal(signal, num_fade_samples):
    fade_in = np.linspace(0, 1, num_fade_samples)
    fade_out = np.linspace(1, 0, num_fade_samples)
    signal[:num_fade_samples] *= fade_in
    signal[-num_fade_samples:] *= fade_out
    return signal


def get_segment_time(
        pcg: np.ndarray, 
        fs_old: float, 
        fs_new: float, 
        label: str,
        class_counts: dict[str, int],
        total_labels: int,
        time: float = 1.25,
        num_fragments: float = 20,
        start_padding: float = 0.3
) -> list[list[int]]:
    """Gets the PCG segments based on time."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old) if fs_old != fs_new else pcg

    initial_num_fragments = num_fragments
    start_idx = round(fs_new * start_padding)
    sig_len = int(len(pcg_resampled) - start_padding)
    num_fragments = int(max(((total_labels - class_counts[label]) * num_fragments) / class_counts[label], num_fragments))
    segment_len = round(fs_new * (time))
    makeup_fragments = max(int(num_fragments - (sig_len / segment_len)), 0)
    overlap = math.ceil((segment_len * makeup_fragments) / (num_fragments)) # 2 for the double overlap on most segments
    print(f"{num_fragments=}, {segment_len=}, {overlap=}, {makeup_fragments=}, {initial_num_fragments=}, {time=}, {fs_new=}, {sig_len=}")

    sample_increment = segment_len - overlap

    seg_idxs = [[i, i, i, i] for i in range(start_idx, len(pcg_resampled), sample_increment)]

    return seg_idxs


def get_hand_label_seg_pcg(path: str, filename: str) -> list[list[Any]]:

    segment_info = loadmat(os.path.join(path, f"{filename}"))
    segment_info = segment_info['state_ans']

    # Remember to adjust index for python instead of matlab
    breakpoint()

    return segment_info


def get_segment_pcg(pcg: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """Gets the PCG segments using mixture of MATLAB and python."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old) if fs_old != fs_new else pcg # type: ignore

    pcg_resampled = ENG.butterworth_low_pass_filter(pcg_resampled, 2, 400, fs_new) # type: ignore
    pcg_resampled = ENG.butterworth_high_pass_filter(pcg_resampled, 2, 25, fs_new) # type: ignore
    pcg_resampled = np.array(pcg_resampled).reshape(-1, 1)
    pcg_resampled = ENG.schmidt_spike_removal(pcg_resampled, float(fs_new))  # type: ignore

    assigned_states = ENG.segmentation(pcg_resampled, fs_new) # type: ignore
    seg_idxs = np.asarray(ENG.get_states(assigned_states), dtype=int) - 1 # type: ignore

    return seg_idxs


def resample(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    return ssg.resample_poly(signal, fs_new, fs_old)


def low_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="lowpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def high_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="highpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def delay_signal(signal: np.ndarray, delay: int) -> np.ndarray:
    """
       Delays a signal by the specified delay
    """
    hh = np.concatenate((
        np.zeros(delay),
        np.ones(1),
        np.zeros(delay)),
        dtype="float32"
    )

    delayed_signal = np.asarray(ssg.lfilter(hh.flatten(), 1, signal))

    return delayed_signal


def correlations(xdn: np.ndarray, ydn: np.ndarray, FL: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
        Calculates the correlation matrix and crosscorrelation vector
    """
    DL = max(np.shape(xdn))
    RXX: np.ndarray = np.zeros((FL, FL), dtype="float32")
    rxy: np.ndarray = np.zeros((FL, 1), dtype="float32")
    ryy: float = 0

    yp: np.ndarray = np.zeros(DL, dtype="float32")
    for ii in range(FL, DL, 1):
        xv = xdn[ii:ii-FL:-1].reshape(-1, 1)
        RXX = RXX + xv @ xv.T
        rxy = rxy + xv * ydn[ii]
        ryy = ryy + ydn[ii] ** 2
        yp[ii] = ydn[ii]

    return RXX, rxy, ryy, yp


def multi_correlations(vdn: np.ndarray, xdn: np.ndarray, FL: int) -> tuple:
    DL, M = vdn.shape  # Number of samples and channels

    # Initialize matrices
    RVV = np.zeros((M * FL, M * FL), dtype="float64")
    Rvx = np.zeros((M * FL, M), dtype="float64")
    rxx = np.zeros((M, M), dtype="float64")
    xp = np.zeros((M, DL), dtype="float64")

    for ii in range(FL, DL):
        # Stack the past FL samples for all channels into a vector
        v_vec = vdn[ii:ii-FL:-1, :].reshape(-1, order='F')  # Shape: (M * FL, )
        x_vec = xdn[ii, :].reshape(-1, 1)  # Shape: (M, 1)

        # Update RVV
        RVV += np.outer(v_vec, v_vec)
        # Update Rvx
        Rvx += v_vec[:, np.newaxis] @ x_vec.T
        # Update rxx
        rxx += x_vec @ x_vec.T
        # Update xp
        xp[:,ii] = xdn[ii, :] 

    # Normalize by the number of samples
    num_samples = DL - FL
    RVV /= num_samples
    Rvx /= num_samples
    rxx /= num_samples

    return RVV, Rvx, rxx, xp


def optimal_weights(RXX: np.ndarray, rxy: np.ndarray, ryy: float | np.ndarray, FL: int, DL: float) -> np.ndarray:
    """
        Finds the optimal weights
    """
    err0 = 0.0005
    egv = np.linalg.eigvals(RXX)
    egv = egv[np.argmax(np.abs(egv))]

    # FIXME: Pre compute the inverse thing to save computations
    # Define calcs as lambdas
    err_fun = lambda w, beta : np.trace(((ryy - 2 * w.T @ rxy + w.T @ RXX @ w + beta * egv * w.T @ w) / (DL - FL)))
    w_fun = lambda err0 : np.linalg.lstsq(RXX + err0 * (egv) * np.eye(RXX.shape[0]), rxy, rcond=-1)[0]
    logging.info("Calculated w_fun")

    w = w_fun(err0)
    err = err_fun(w, err0)

    total_passes = 0
    passes = 0
    err_prev = 0
    while abs(err0 - err) > 1e-4 or passes <= 2:
        if err == err_prev:  # To get the same result as matlab.
            passes += 1
        w = w_fun(err0)

        err_prev = err0
        err0 = err

        err = err_fun(w, err0)

        total_passes += 1
        logging.info(f"Weiner filter, {total_passes=}")
        if total_passes > 150:
            return w

    return w


def multi_optimal_weights(RVV: np.ndarray, rvx: np.ndarray, rxx: np.ndarray, FL: int, DL: float, M: int) -> np.ndarray:
    """
        Finds the optimal weights
    """
    err0 = 0.0005
    egv = np.linalg.eigvals(RVV)
    egv = float(egv[np.argmax(np.abs(egv))])
    I_ML = np.eye(RVV.shape[0])

    mse = lambda w, beta : np.trace((rxx - 2 * w.T @ rvx + w.T @ RVV @ w + beta * egv * w.T @ w) / (DL - FL))
    weiner_hopf = lambda beta : np.linalg.inv(RVV + beta * egv * I_ML) @ rvx

    w = weiner_hopf(err0)
    err = mse(w, err0)
    logging.info("Calculated w_fun")

    total_passes = 0
    passes = 0
    err_prev = 0
    while (abs(err0 - err) > 0.0001) or passes <= 2:
        if (abs(err - err_prev) < 0.0000001):  # To get the same result as matlab.
            passes += 1
        w = weiner_hopf(err0)
        
        err_prev = err0
        err0 = err
        err = mse(w, err0)

        total_passes += 1
        logging.info(f"Weiner filter, {total_passes=}")
        if total_passes > 150:
            return w

    return w


def weiner_filter(xdn: np.ndarray, ydn: np.ndarray, FL: int, DL: float) -> np.ndarray:
    """
        Runs the weiner filter algorithm
    """
    RXX, rxy, ryy, yp = correlations(xdn, ydn, FL)
    w = optimal_weights(RXX, rxy, ryy, FL, DL)

    # apply weiner filter
    yhat = ssg.lfilter(w.flatten(), 1, xdn) # w^T v
    #yhat2 = ssg.fftconvolve(w.flatten(), xdn, mode='valid')
    #print(yhat - yhat2) 
    e = yp.T - yhat # e = x - w^T v

    return e, yhat, w, RXX, rxy, ryy


def multi_weiner_filter(vdn: np.ndarray, xdn: np.ndarray, FL: int, DL: int, fs: int) -> np.ndarray:
    """
    Computes the Wiener filters between all the different channels and builds the matrix.
    """
    N, M = vdn.shape  # Number of samples and channels
    W = np.zeros((M * FL, M), dtype="float64")

    RVV, Rvx, rxx, xp = multi_correlations(vdn, xdn, FL)
    #RVV, Rvx, rxx, xp, _, _, _, _ = both_correlations(vdn, xdn, FL)
    W = multi_optimal_weights(RVV, Rvx, rxx, FL, DL, M).astype("float64")


    xhat = np.zeros((N, M), dtype="float64")

    for channel in range(M):
        #xhat[:, channel] = ssg.lfilter(W[:, channel].flatten(), 1, vdn[:, channel])
        for sub_channel in range(M):
            # Get the filter coefficients for the sub_channel and channel
            filter_coeffs = W[sub_channel * FL:(sub_channel + 1) * FL, channel]
            # Apply the filter to the sub_channel
            filtered = ssg.lfilter(filter_coeffs.flatten(), 1, vdn[:, sub_channel])
            # Accumulate the filtered signal for the current channel
            xhat[:, channel] += filtered

    e = xp - xhat.T
    return e, xhat.T


def noise_canc(xdn: np.ndarray, ydn: np.ndarray, fc: float = 150, fs: float = 1000, FL:int = 128, hp: bool = False) -> np.ndarray:
    """
    Noise cancellation using weiner filter and hp

    xdn is background noise,
    ydn is the signal with background noise
    """
    DL = max(np.shape(xdn))
    ydn = delay_signal(ydn, math.floor(FL/2)) 

    # High pass if required due to small filter length
    if hp:
        xdn = high_pass_butter(xdn, 2, fc, fs)

    return weiner_filter(xdn, ydn, FL, DL)


def multi_noise_canc(vdn: np.ndarray, xdn: np.ndarray, fc: float = 150, fs: float = 1000, FL: int = 128, delay_offset: int = 0) -> np.ndarray:
    """
    Noise cancellation of multichannel signals using the Wiener filter.
    """
    DL, _ = vdn.shape
    for channel in range(min(xdn.shape)):
        xdn[:, channel] = delay_signal(xdn[:, channel], (FL) // 2 + delay_offset)
    return multi_weiner_filter(vdn, xdn, FL, DL, int(fs))


def wavelet_denoise(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    signal = standardise_signal(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6725
    var = np.var(coeffs[-1])
    threshold = sigma**2 / np.sqrt(max(var - sigma**2, 1e-30))

    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, wavelet)


def wdenoise(signal: np.ndarray, wavelet: str, level: int, method: Optional[str] = None):
    if method is not None:
        denoised_signal = ENG.wdenoise(signal, float(level), 'Wavelet', wavlet, 'DenoisingMethod', method) # type: ignore
    else:
        denoised_signal = ENG.wdenoise(signal, float(level), 'Wavelet', wavelet) # type: ignore

    return denoised_signal


def add_chirp(audio_signal, fs):
    t = np.arange(len(audio_signal)) / fs

    chirp_signal = scipy.signal.chirp(t, f0=0, f1=fs/2, t1=t[-1], method='linear')
    chirp_signal = (chirp_signal / np.max(np.abs(chirp_signal))) * max(0.5, np.max(np.abs(audio_signal)))

    return audio_signal + chirp_signal

def create_spectrogram(signal, transform):
    spectrogram = transform(signal)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    return spectrogram