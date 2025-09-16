"""
    fileio.py
    Author : Milan Marocchi

    Reading and writing records
"""

import wfdb
import json
from json import JSONEncoder
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import scipy.io as sio


def get_ticking_channel_map(collection: int) -> dict:
    if collection == 1:
        channel_map = {
            '1': 1,
            '2': 2,
            '3': 4,
            '4': 5,
            '5': 6,
            '6': 7,
            '7': 8,
            'E': 11,
        }
    elif collection == 2:
        channel_map = {
            '1': 2,
            '2': 4,
            '3': 5,
            '4': 6,
            '5': 8,
            '6': 7,
            '7': 9,
            'E': 11,
            'E2': 3,
        }
    # This collection represents the processed vest_data channel format
    elif collection == -1:
        channel_map = {
            '1': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '5': 4,
            '6': 5,
            '7': 6,
            'E': 7,
            'E2': 8,
        }
    else:
        raise ValueError(f"Incorrect collection number: {collection=}")

    return channel_map


def read_vest_data(filename: str, max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Read the processed vest data in its standardised form."""
    signal, fs = read_signal_wav(filename)
    return signal[:max_len*fs, :], fs


def get_vest_record(path):
    signal, fs = read_signal_wav(path)

    try:
        output = {
            '1': signal[:, 0],
            '2': signal[:, 1],
            '3': signal[:, 2],
            '4': signal[:, 3],
            '5': signal[:, 4],
            '6': signal[:, 5],
            '7': signal[:, 6],
            'E': signal[:, 8],
            'E2': signal[:, 9],
        }
    except IndexError as e:
        output = {
            '1': signal[:, 0],
            '2': signal[:, 1],
            '3': signal[:, 2],
            '4': signal[:, 3],
            '5': signal[:, 4],
            '6': signal[:, 5],
            '7': signal[:, 6],
            'E': signal[:, 8],
        }

    return output, fs


def read_ticking_PCG(filename: str, channel: int, noise_mic: bool = False, collection: int = 1, max_len: Optional[int] = None, all_channels=False) -> Tuple[np.ndarray, int]:
    """Read in the PCG from the ticking heart data."""
    channels = get_ticking_channel_map(collection)

    if ".wav" not in filename:
        filename += ".wav"

    signal, fs = read_signal_wav(filename)
    wav_channel = channels[str(channel)]
    wav_channel = wav_channel + 7 if noise_mic else wav_channel
    wav_channel = wav_channel -1 if collection != -1 else wav_channel # -1 for MATLAB indexs, otherwise would be + 8 for NM
    return signal[:, wav_channel], fs

def read_ticking_channels(filename: str) -> Tuple[np.ndarray, int]:
    """Reads in all the vest-data in the processed channel map configuration"""
    signal, fs = read_signal_wav(filename)
    assert min(signal.shape) == 8 or min(signal.shape) == 9, "Only for reading the processed vest-data"
    return signal, fs

def read_ticking_ECG(filename: str, collection: int = 1, channel: int = 1, max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Read in the ECG from the ticking heart data."""
    ecg_channel = 'E' if channel == 1 else 'E2'
    ecg = get_ticking_channel_map(collection)[ecg_channel]
    ecg = ecg - 1 if collection != -1 else ecg # -1 for MATLAB indexs
    filename += ".wav"
    signal, fs = read_signal_wav(filename)

    return signal[:, ecg], fs


def read_signal_wav(filename: str) -> Tuple[np.ndarray, int]:
    """
    Reads in a signal from a wav file then converts it into the same format that matlab would output.
    Outputs the sampling freq as well as the signal
    """
    if ".wav" not in filename:
        filename += ".wav"

    Fs, signal = sio.wavfile.read(filename)

    if signal.dtype == np.int16:
        max_val = np.iinfo(np.int16).max
    elif signal.dtype == np.int32:
        max_val = np.iinfo(np.int32).max
    elif signal.dtype == np.int64:
        max_val = np.iinfo(np.int64).max
    elif signal.dtype == np.float32 or signal.dtype == np.float64:
        return signal.astype(np.float32), Fs
    else:
        raise ValueError("Unsupported data type")

    # Convert to float 32
    signal = (signal / max_val).astype(np.float32)

    return signal, Fs


def save_signal_wav(signal: np.ndarray, fs: int, path: str):
    """
    Saves a signal as a wav file to the specified path.
    """
    if ".wav" not in path:
        path += ".wav"

    sio.wavfile.write(path, fs, signal)


def get_cinc_record(path: str, max_len: Optional[int] = None) -> wfdb.Record:

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_len is None:
        target_sig_len = sig_len
    else:
        target_sig_len = min(round(max_len * fs), sig_len) # type: ignore

    rec = wfdb.rdrecord(path, sampfrom=0, sampto=target_sig_len)
    return rec


def get_cinc_sig(path: str, name: str, max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:

    record = get_cinc_record(path, max_len)
    fs = int(record.fs) # type: ignore
    signal = record.p_signal[:, record.sig_name.index(name)] # type: ignore

    return signal , fs


def get_cinc_ecg(path: str, max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Read in the ECG data from the cinc dataset, without inserting Nan values 
    """

    FS = 2000 # ECG is recorded at 2000Hz

    if ".dat" not in path:
        path += ".dat"

    with open(path, 'rb') as fid:
        ecg = np.fromfile(fid, dtype=float)

    last_index = round(max_len * FS) if max_len is not None else len(ecg)

    return ecg, FS
    

def get_cinc_pcg(path: str, max_len: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Read in the PCG data from the cinc dataset, without inserting Nan values
    """

    if ".wav" not in path:
        path += ".wav"

    pcg, Fs = read_signal_wav(path)

    last_index = round(max_len * Fs) if max_len is not None else len(pcg)

    return pcg, Fs


def get_patients(path: str, training_a: bool = False) -> Tuple[list[str], list[str]]:
    """
    Gets the labels from the cinc data
    """
    training_exclude = ['a0041', 'a0117', 'a0220', 'a0233']

    # Check to see if there are headers 
    with open(path, 'r') as file_in:

        line = file_in.readline()
        if line[0] == "#":
            line = file_in.readline()

        if 'patient' in line and 'abnormality' in line:
            patients_header = 'patient'
            label_header = 'abnormality'
            header = 0
        elif 'patient' in line and 'recording' in line and 'diagnosis' in line:
            patients_header = 'patient'
            label_header = 'diagnosis'
            header = 0
        elif 'subject' in line and 'abnormality' in line:
            patients_header = 'subject'
            label_header = 'abnormality'
            header = 0
        else:
            patients_header = 0
            label_header = 1
            header = None

    patient_data = pd.read_csv(path, comment='#', header=header)
    patients = list(patient_data[patients_header])
    labels = list(patient_data[label_header])

    patients = [patient for patient in patients if patient not in training_exclude]

    return patients, labels

def get_patients_recordings(path: str) -> Tuple[list[str], list[str], list[str]]:
    """
    Gets the labels from the vest data (ticking heart)
    """

    # Check to see if there are headers 
    with open(path, 'r') as file_in:

        line = file_in.readline()
        if line[0] == "#":
            line = file_in.readline()

        if 'patient' in line and 'abnormality' in line and 'recording' in line:
            patients_header = 'patient'
            label_header = 'abnormality'
            recording_header = 'recording'
            header = 0
        elif 'patient' in line and 'diagnosis' in line and 'recording' in line:
            patients_header = 'patient'
            label_header = 'diagnosis'
            recording_header = 'recording'
            header = 0
        else:
            patients_header = 0
            label_header = 1
            header = None

    patient_data = pd.read_csv(path, comment='#', header=header)
    patients = list(patient_data[patients_header])
    recordings = list(patient_data[recording_header])
    labels = list(patient_data[label_header])

    return patients, recordings, labels



def get_patients_split(path: str, subset: str, fold: int = 1) -> pd.DataFrame:
    """
    Gets the label data from the splits file
    """
    patients = pd.read_csv(path, comment='#')
    split_name = 'split' if fold == 1 else f'split{fold}'
    patients = patients[patients[split_name] == subset]

    return patients


def get_patients_segments(path: str) -> Tuple[list, int, int]:
    """
    Gets the segment information from the patient file
    """
    data = read_json_numpy(path)

    return data['segments'], data['last_index'], data['fs']


class NumpyEncoder(JSONEncoder):
    """
    Class to encode numpy data to a list to be stored in a json file.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        # you can add more conversions if needed (e.g. np.bool_)
        return super().default(o)


def write_json_numpy(data: str, filepath: str):
    """
    Writes to a json using the numpy encoder
    """
    # NOTE: Add error checking so format is enforced.

    with open(filepath, "w") as out_file:
        json.dump(data, out_file, cls=NumpyEncoder)


def read_json_numpy(filepath: str) -> dict:
    """
    Reads from a json file and decodes the array to a numpy array.
    Excepts the format that is used when written
    """
    if ".json" not in filepath:
        filepath += ".json"

    with open(filepath, "r") as in_file:
        json_data = json.load(in_file)

    return json_data

def save_segment_info(path, segment_info):
    with open(path, 'w') as f:
        json.dump(segment_info, f, cls=NumpyEncoder)


def load_segment_info(path):
    with open(path, 'r') as f:
        segment_info = json.load(f)
    return segment_info