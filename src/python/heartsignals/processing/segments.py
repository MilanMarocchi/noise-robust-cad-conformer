"""
    segments.py
    Author: Leigh Abbott

    Purpose: Segment the signals into their heart segments
"""

import random
from typing import Tuple
import numpy as np
import scipy

from processing.process import normalise_2d_array_length, normalise_array_length
from util.fileio import read_json_numpy
from util.utils import first_dict_value

def get_patients_segments(path: str) -> Tuple[list, int, int]:
    """
    Gets the segment information from the patient file
    """
    data = read_json_numpy(path)

    return data['segments'], data['last_index'], data['fs']

def resample_seg_joins(join_idxs, old_fs, new_fs):

    if new_fs == old_fs:
        return join_idxs.copy(), old_fs

    assert new_fs > old_fs and new_fs % old_fs == 0, f'{new_fs=}, {old_fs=}'

    def _resample(idx):
        return round(idx * (new_fs / old_fs))

    join_idxs = join_idxs.copy()
    for idx_idx, (join_idx_start, join_idx_end) in enumerate(join_idxs):
        join_idxs[idx_idx] = (_resample(join_idx_start), _resample(join_idx_end))

    return join_idxs, new_fs

def resample_seg_joins_norm(join_idxs, old_fs, new_fs):

    if new_fs == old_fs:
        return join_idxs.copy(), old_fs

    assert new_fs > old_fs and new_fs % old_fs == 0, f'{new_fs=}, {old_fs=}'

    def _resample(idx):
        return round(idx * (new_fs / old_fs))

    join_idxs = join_idxs.copy()
    for idx_idx, join_idx in enumerate(join_idxs) :
        join_idxs[idx_idx] = _resample(join_idx)

    return join_idxs, new_fs

def resample_segments(seg_idxs: list, old_fs: float, new_fs: float) -> tuple[list, float]:

    adj_seg_idxs = seg_idxs.copy()

    if old_fs != new_fs:

        for i in range(len(seg_idxs)):
            for j in range(len(seg_idxs[i])):
                adj_seg_idxs[i][j] = round(adj_seg_idxs[i][j] * (new_fs / old_fs))

    return adj_seg_idxs, new_fs


def get_shortest_signal_len(signals_dict: dict[str, np.ndarray]) -> int:
    return min(len(sig) for sig in signals_dict.values())


def crop_signal_ends(signal: np.ndarray, crop_start: int, crop_end: int) -> np.ndarray:
    return signal[crop_start:crop_end]


def shift_seg_idxs(seg_idxs: np.ndarray, shift: int) -> list[list]:
    return [[idx - shift for idx in idx_group]
            for idx_group in seg_idxs]


def crop_seg_idxs(seg_idxs: np.ndarray, crop_start: int, crop_end: int) -> list:

    adj_seg_idxs = []

    shift_idxs = shift_seg_idxs(seg_idxs, crop_start)

    for seg_idx_group in shift_idxs:
        if all(0 <= idx < crop_end for idx in seg_idx_group):
            adj_seg_idxs.append(seg_idx_group)

    return adj_seg_idxs


def crop_sigs_segs(signals_dict: dict[str, np.ndarray], crop_start: int, crop_end: int, seg_idxs: list = list()) -> tuple[dict[str, np.ndarray], list]:

    start = crop_start
    shortest_len = min([len(sig) for _, sig in signals_dict.items()])
    end = shortest_len - crop_end

    for sig_name, sig in signals_dict.items():
        old_len = len(signals_dict[sig_name])
        signals_dict[sig_name] = sig[start:end]
        new_len = len(signals_dict[sig_name])
        assert new_len == shortest_len - (crop_start + crop_end), f'{old_len=}, {new_len=}, {crop_end=}'

    adj_seg_idxs = list()
    if len(seg_idxs) > 0:
        adj_seg_idxs = list()

        for seg_idx in seg_idxs:
            adj_seg_idx = [idx - start for idx in seg_idx]
            if all(0 <= idx < end for idx in adj_seg_idx):
                adj_seg_idxs.append(adj_seg_idx)

    return signals_dict, adj_seg_idxs


def from_matform(matform: list) -> list:
    sample_points = []

    latest_state = None

    for s in matform[0:-1]:
        sample_num = s[0][0][0] - 1
        state = s[1][0]

        assert state in ['S1', 'systole', 'S2', 'diastole', '(N', 'N)'], f'{state=}'
        if state == 'S1':
            if latest_state is not None:
                sample_points.append(latest_state)
            latest_state = [sample_num]
        elif state == '(N':
            latest_state = None
        elif latest_state is not None:
            latest_state.append(sample_num)
        else:
            continue

    return sample_points


def get_seg_offset(seg_idxs: list, seg_ind: int) -> int:

    num_cycles = len(seg_idxs) - 1
    num_seg_idx = seg_idxs[num_cycles][seg_ind] - seg_idxs[0][seg_ind]
    seg_offset = round(0.25 * (num_seg_idx / num_cycles))

    return seg_offset


def get_heart_sound_durations(seg_idxs: list) -> tuple[list, list]:

    s1_i, sys_i, s2_i, dia_i = 0, 1, 2, 3

    s1_durs = []
    s2_durs = []

    for idx_group in seg_idxs:

        s1_s, sys_s, s2_s, dia_s = idx_group[s1_i], idx_group[sys_i], idx_group[s2_i], idx_group[dia_i]

        s1_durs.append(sys_s - s1_s)
        s2_durs.append(dia_s - s2_s)

    return s1_durs, s2_durs


def create_segment_waveform(seg_idxs: list, signal_len: int) -> np.ndarray:

    s1_i, sys_i, s2_i, dia_i = 0, 1, 2, 3

    seg_signal = np.zeros(signal_len)

    for idx_group in seg_idxs:

        s1_s, sys_s, s2_s, dia_s = idx_group[s1_i], idx_group[sys_i], idx_group[s2_i], idx_group[dia_i]

        seg_signal[s1_s:sys_s] = s1_i
        seg_signal[sys_s:s2_s] = sys_i
        seg_signal[s2_s:dia_s] = s2_i
        seg_signal[dia_s:] = dia_i

    return seg_signal

def seg_idxs_from_mat(matform):
    seg_joins = []

    curr_group_end = next_group_start = segment_start = None

    for s in matform:

        state = s[1][0]
        sample_num = s[0][0][0] - 1

        if state == 'diastole':
            curr_group_end = sample_num
        elif state == 'S1':
            next_group_start = sample_num
        elif state == '(N':
            curr_group_end = next_group_start = segment_start = None
        elif state == 'N)':
            continue

        if curr_group_end is not None and next_group_start is not None:
            segment_end = round((curr_group_end + next_group_start) / 2)

            if segment_start is not None:
                seg_joins.append((segment_start, segment_end))

            segment_start, curr_group_end, next_group_start = segment_end, next_group_start, None

    return seg_joins


def get_seg_time_join_idx(seg_idxs: list, last_len: int) -> list:

    num_groups = len(seg_idxs)
    seg_joins = list()

    for i in range(num_groups-1):
        join = seg_idxs[i][0]
        seg_joins.append(join)

    return seg_joins


def get_seg_join_idx(seg_idxs: list, last_len: int) -> list:

    seg_ind = 0
    num_ind = len(seg_idxs[0])

    seg_joins = []
    num_groups = len(seg_idxs)

    for i in range(num_groups-1):
        curr_group_end = seg_idxs[i][seg_ind]
        next_group_start = seg_idxs[(i + 1)][seg_ind]

        if i == num_groups - 1:
            assert next_group_start == 0, f'{next_group_start=}'
            next_group_start += last_len

        if curr_group_end < 0:
            continue

        assert next_group_start > curr_group_end > 0, f'{curr_group_end=}, {next_group_start=}'

        join = curr_group_end
        seg_joins.append(join)

    return seg_joins


# THE LEIGH SPECIAL
def leigh_get_seg_join_idx(seg_idxs: list, last_len: int) -> list:

    seg_ind = 3
    num_ind = len(seg_idxs[0])

    seg_joins = []
    num_groups = len(seg_idxs)

    for i in range(num_groups-1):
        curr_group_end = seg_idxs[i][seg_ind]
        next_group_start = seg_idxs[(i + 1)][(seg_ind + 1) % num_ind]

        if i == num_groups - 1:
            # assert next_group_start == 0, f'{next_group_start=}'
            next_group_start += last_len

        if curr_group_end < 0:
            continue

        assert next_group_start > curr_group_end > 0, f'{curr_group_end=}, {next_group_start=}'

        join = round((curr_group_end + next_group_start) / 2)
        seg_joins.append(join)

    return seg_joins


def segment_data(data: np.ndarray, segment_joins: np.ndarray, fs: int, start_offset: float = 0.0, overlap: float = 0.25, segment_type: str = 'time', max_len: float = 2) -> np.ndarray:

    # Turn 1d to 2d, multiple x long fragments. Only support time fow now
    if segment_type != 'time':
        raise ValueError(f"{segment_type}: is not currently supported")

    segmented_data = []

    for idx, segment in enumerate(segment_joins):
        start_idx = fs * start_offset if idx == 0 else segment_joins[idx - 1] - idx * (fs * overlap)
        end_idx = segment + (fs * start_offset) if segment + 300 < len(data) else len(data) 

        if data.ndim > 1 and min(data.shape) > 1:
            normalised_data, _ = normalise_2d_array_length(data[int(start_idx):int(end_idx), :], int(max_len * fs))
        else:
            normalised_data, _ = normalise_array_length(data[int(start_idx):int(end_idx)], int(max_len * fs))
        segmented_data.append(normalised_data)

    segmented_data = np.asarray(segmented_data)
    return segmented_data


def stretch_signal(signal, spline=False):

    len_sig = len(signal)

    original_indices = np.linspace(0, len_sig - 1, len_sig)

    stretched_indices = np.linspace(0, len_sig - 1, 2 * len_sig)

    if spline:
        stretched_signal = scipy.interpolate.UnivariateSpline(original_indices, signal, s=0)(stretched_indices)
    else:
        stretched_signal = np.interp(stretched_indices, original_indices, signal)

    return stretched_signal


def spectral_smoothing(signal, alpha=0.5):
    S = np.abs(np.fft.rfft(signal))
    smoothed_S = np.zeros_like(S)
    smoothed_S[0] = S[0]
    for i in range(1, len(S)):
        smoothed_S[i] = alpha * S[i] + (1 - alpha) * smoothed_S[i-1]
    phase = np.angle(np.fft.rfft(signal))
    smoothed_spectrum = smoothed_S * np.exp(1j * phase)
    return np.fft.irfft(smoothed_spectrum)


def crossfade_and_concatenate_hanning(curr_signal, next_signal, num_fade_samples, interpolate_fade=True):
    if num_fade_samples == 0:
        return np.hstack((curr_signal, next_signal))

    fade_out = np.hanning(2 * num_fade_samples)[:num_fade_samples]
    fade_in = np.hanning(2 * num_fade_samples)[num_fade_samples:]

    crossfaded_segment = (curr_signal[-num_fade_samples:] * fade_out +
                          next_signal[:num_fade_samples] * fade_in)

    if interpolate_fade:
        crossfaded_segment = stretch_signal(crossfaded_segment, spline=True)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_equal_voltage(curr_signal, next_signal, num_fade_samples, interpolate_fade=True):

    if num_fade_samples == 0:
        return np.hstack((curr_signal, next_signal))

    fade_out = np.linspace(1, 0, num_fade_samples)
    fade_in = np.linspace(0, 1, num_fade_samples)

    crossfaded_segment = (curr_signal[-num_fade_samples:] * fade_out +
                          next_signal[:num_fade_samples] * fade_in)

    if interpolate_fade:
        crossfaded_segment = stretch_signal(crossfaded_segment)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_equal_power(curr_signal, next_signal, num_fade_samples, interpolate_fade=True):

    if num_fade_samples == 0:
        return np.hstack((curr_signal, next_signal))

    t = np.linspace(0, np.pi / 2, num_fade_samples)

    fade_out_curve = np.cos(t)
    fade_in_curve = np.sin(t)

    crossfaded_segment = (curr_signal[-num_fade_samples:] * fade_out_curve +
                          next_signal[:num_fade_samples] * fade_in_curve)

    if interpolate_fade:
        crossfaded_segment = stretch_signal(crossfaded_segment, spline=True)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_corr(curr_signal, next_signal, num_fade_samples, interpolate_fade=True):

    # https://music.columbia.edu/pipermail/music-dsp/2011-July/069971.html

    if num_fade_samples == 0 or num_fade_samples == 1:
        return np.hstack((curr_signal, next_signal))

    curr_signal_cut = curr_signal[-num_fade_samples:]
    next_signal_cut = next_signal[:num_fade_samples]

    # linear crossfade
    if abs(np.var(curr_signal_cut)) < 1E-5 or abs(np.var(next_signal_cut)) < 1E-5:
        return crossfade_and_concatenate_equal_voltage(curr_signal, next_signal,
                                                       num_fade_samples, interpolate_fade)
    r = np.abs(np.corrcoef(curr_signal_cut, next_signal_cut)[0, 1])

    assert not np.isnan(r).any(), f'{r=}, {curr_signal_cut=}, {next_signal_cut}'

    t = np.linspace(-1, 1, num_fade_samples)

    # Flattened Hann crossfade
    o = np.piecewise(t, [t < -1, (t >= -1) & (t < 1), t >= 1],
                     [lambda t: 0.5 * np.sign(t),
                      lambda t: (9/16)*np.sin(np.pi/2 * t) + (1/16)*np.sin(3*np.pi/2 * t),
                      lambda t: 0.5 * np.sign(t)])

    e = np.sqrt((0.5 / (1 + r)) - ((1 - r) / (1 + r)) * o**2)

    a = e + o

    fade_out = 1 - a
    fade_in = a

    crossfaded_segment = (curr_signal_cut * fade_out +
                          next_signal_cut * fade_in)

    if interpolate_fade:
        crossfaded_segment = stretch_signal(crossfaded_segment, spline=True)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def compute_entropy(psd):
    normalized_psd = psd / np.sum(psd)
    entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-8))  # added small value to avoid log(0)
    return entropy


def get_seg_info(seg_idxs):

    seg_ind = 0
    seg_offset = get_seg_offset(seg_idxs, seg_ind)
    adj_seg_idx_starts = []

    for seg in seg_idxs:
        adj_seg = seg[seg_ind] - seg_offset
        if adj_seg > 0:
            adj_seg_idx_starts.append(adj_seg)

    seg_info = {
        'orig_idx_segs': seg_idxs,
        'seg_ind': seg_ind,
        'adj_seg_idx_starts': adj_seg_idx_starts,
        'num_adj_idx': len(adj_seg_idx_starts),
        'num_segs': len(adj_seg_idx_starts) - 1,
    }

    return seg_info


def filter_segments(multi_segments_dict, target_signal, sample_rate, threshold_factor):

    target_segments = multi_segments_dict[target_signal]

    entropies = []
    for segment in target_segments:
        _, _, Zxx = scipy.signal.stft(segment, fs=sample_rate, nperseg=256)
        power = np.abs(Zxx)**2
        psd = np.sum(power, axis=1)
        entropy = compute_entropy(psd)
        entropies.append(entropy)

    median_entropy = np.median(entropies)
    mad = np.median(np.abs(entropies - median_entropy))  # type: ignore

    upper_bound = median_entropy + threshold_factor * mad
    zero_segments = [entropy > upper_bound for entropy in entropies]

    # multi_segments_dict = {
    #     key: [np.zeros_like(segment) if zero else segment for zero, segment in zip(zero_segments, segments)]
    #     for key, segments in multi_segments_dict.items()
    # }

    multi_segments_dict = {
        key: [segment for zero, segment in zip(zero_segments, segments) if not zero]
        for key, segments in multi_segments_dict.items()
    }

    return multi_segments_dict, len(zero_segments)


def shuffle_segments_multi(multi_segments_dict, group_sizes, num_segments=None):
    num_segments = len(first_dict_value(multi_segments_dict)) if num_segments is None else num_segments

    grouped_indices = []
    i = 0
    size_idx = 0
    while i < num_segments:
        current_group_size = group_sizes[size_idx % len(group_sizes)]
        grouped_indices.append(list(range(i, min(i + current_group_size, num_segments))))
        i += current_group_size
        size_idx += 1

    random.shuffle(grouped_indices)

    permuted_segments_dict = {
        key: [
            np.hstack([val[i]
                       for i in group
                       if i < len(val)])
            for group in grouped_indices
        ] for key, val in multi_segments_dict.items()
    }

    return permuted_segments_dict


def join_adjacent_segments(multi_segments_dict, num_segments=None, random_start=True):
    num_segments = len(first_dict_value(multi_segments_dict)) if num_segments is None else num_segments

    all_indices = list(range(num_segments))

    start_idx = random.randint(0, num_segments - 1) if random_start else 0
    reordered_indices = all_indices[start_idx:] + all_indices[:start_idx]

    permuted_segments_dict = {
        key: [np.hstack([val[i] for i in reordered_indices])]
        for key, val in multi_segments_dict.items()
    }

    return permuted_segments_dict


def shuffle_segments(multi_segments_dict, group_size=4, num_segments=None):
    num_segments = len(first_dict_value(multi_segments_dict)) if num_segments is None else num_segments

    grouped_indices = [list(range(i, i + group_size)) for i in range(0, num_segments, group_size)]

    random.shuffle(grouped_indices)

    permuted_segments_dict = {
        key: [multi_segments_dict[key][i]
              for group in grouped_indices
              for i in group
              if i < len(multi_segments_dict[key])]
        for key in multi_segments_dict.keys()
    }

    return permuted_segments_dict


def shuffle_segments_fy(multi_segments_dict, group_size=1, num_segments=None):
    # uses Fisher-Yates algorithm

    num_segments = len(first_dict_value(multi_segments_dict)) if num_segments is None else num_segments

    all_indices = list(range(num_segments))

    for i in range(len(all_indices) - 1, 0, -1):
        j = random.randint(0, i-1)

        if i // group_size == j // group_size:
            j = i - 1

        all_indices[i], all_indices[j] = all_indices[j], all_indices[i]

    grouped_indices = [all_indices[i:i + group_size] for i in range(0, num_segments, group_size)]

    permuted_segments_dict = {
        key: [multi_segments_dict[key][i] for group in grouped_indices for i in group]
        for key in multi_segments_dict.keys()
    }

    return permuted_segments_dict


def segment_multi_signals(signals_dict, seg_start_idxs):

    multi_segments = {sig_name: [] for sig_name in signals_dict}

    for idx_idx in range(len(seg_start_idxs)-1):
        for sig_name, sig in signals_dict.items():

            start_ind = seg_start_idxs[idx_idx]
            end_ind = seg_start_idxs[idx_idx+1]

            sig = sig[start_ind:end_ind]
            multi_segments[sig_name].append(sig)

    return multi_segments


def segment_signal_tuple(signal, seg_start_end_idxs):
    segments = [
        signal[seg_start_idx:seg_end_idx]
        for seg_start_idx, seg_end_idx in seg_start_end_idxs
    ]
    return segments


def segment_signal(signal, seg_start_idxs):
    segments = [
        signal[seg_start_idxs[idx_idx]:seg_start_idxs[idx_idx+1]]
        for idx_idx in range(len(seg_start_idxs)-1)
    ]
    return segments


def get_weighted_random_choice(max_num):
    nums = [n + 1 for n in range(max_num)]
    weights = [1.0 / n for n in nums]
    weights /= np.sum(weights)
    return np.random.choice(nums, p=weights)


def some_shuffle_segments_fn(orig_segs, num_segs=None):
    num_segs = len(next(iter(orig_segs.values()))) if num_segs is None else num_segs

    # random_chance = random.random()
    # if random_chance < (1 / 3):
    #     shuffled_segments = shuffle_segments_multi(
    #         orig_segs,
    #         group_sizes=[get_weighted_random_choice(num_segs//2) for _ in range(5)],
    #         num_segments=num_segs)
    # elif random_chance < (2 / 3):
    #     shuffled_segments = shuffle_segments_multi(
    #         orig_segs,
    #         group_sizes=[random.randint(1, 4) for _ in range(5)],
    #         num_segments=num_segs)
    # else:
    #     shuffled_segments = shuffle_segments_multi(
    #         orig_segs,
    #         group_sizes=[1],
    #         num_segments=num_segs)

    shuffled_segments = shuffle_segments_multi(
        orig_segs,
        group_sizes=[1],
        num_segments=num_segs)

    return shuffled_segments


def find_max_ind(segments, min_len):

    total_length = 0
    index = 0
    for i in range(len(segments) - 1, -1, -1):
        total_length += len(segments[i])
        if total_length >= min_len:
            index = i
            break

    return index


def build_signal(segments, num_fade_samples, target_num_samples, num_segs=None):
    num_segs = len(segments) if num_segs is None else num_segs
    num_audio_samples = 0
    idx = 0
    num_joined_segs = 0
    built_signal = []

    while num_audio_samples < target_num_samples:
        sig_seg = segments[idx]

        if num_joined_segs == 0:
            built_signal = sig_seg

            assert built_signal.shape == sig_seg.shape, f'{built_signal.shape=}, {sig_seg.shape}'

        else:
            old_len = len(built_signal)
            built_signal = crossfade_and_concatenate_corr(
                built_signal, sig_seg,
                num_fade_samples=num_fade_samples,
                interpolate_fade=True,
            )
            new_len = len(built_signal)
            assert new_len == (old_len + len(sig_seg)), f'{new_len=}, {old_len=}, {len(sig_seg)=}'

        num_audio_samples += len(sig_seg)
        num_joined_segs += 1
        idx = (idx + 1) % num_segs

    assert len(built_signal) == num_audio_samples >= target_num_samples, f'{num_audio_samples=}, {len(built_signal)=}'

    return built_signal


def plot_signals_with_joins(original_segments_dict, crossfaded_signals_dict, crossfade_join_points):

    num_signals = len(original_segments_dict)
    fig, axs = plt.subplots(num_signals, 2, figsize=(18, 6*num_signals), sharex='col')

    if num_signals == 1:
        axs = axs.reshape((1, -1))  # Ensure axs is always a 2D array for consistency

    for i, (title, original_segments) in enumerate(original_segments_dict.items()):
        # Concatenate original segments and calculate join points
        original_signal = np.concatenate(original_segments)
        original_join_points = np.cumsum([len(segment) for segment in original_segments[:-1]])

        # Plot original signal
        axs[i, 0].plot(original_signal, label='Original Signal', alpha=0.7)

        # Indicate the join points
        for join in original_join_points:
            axs[i, 0].axvline(x=join, color='r', linestyle='--',
                              label='Original Join Point' if join == original_join_points[0] else '')

        axs[i, 0].set_title(f'Original Segments: {title}')
        # axs[i, 0].set_xlabel('Sample Index')
        axs[i, 0].set_ylabel('Amplitude')
        # axs[i, 0].grid(True)

        # Plot crossfaded signal
        crossfaded_signal = crossfaded_signals_dict[title]
        axs[i, 1].plot(crossfaded_signal, label='Crossfaded Signal', alpha=0.7)

        # Indicate the join points
        for join in crossfade_join_points:
            axs[i, 1].axvline(x=join, color='r', linestyle='--',
                              label='Crossfade Join Point' if join == crossfade_join_points[0] else '')

        axs[i, 1].set_title(f'Crossfaded Signal: {title}')
        # axs[i, 1].set_xlabel('Sample Index')
        axs[i, 1].set_ylabel('Amplitude')
        # axs[i, 1].grid(True)

        # assert original_signal.shape == crossfaded_signal.shape,
        # f'{original_signal.shape=}, {crossfaded_signal.shape=}'

    fig.tight_layout()
    fig.show()