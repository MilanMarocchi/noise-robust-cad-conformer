import logging
from typing import Optional
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
import torch

from tqdm.auto import tqdm
from processing.filtering import (
    normalise_signal, 
    resample
)
from processing.process import (
    normalise_2d_array_length, 
    pre_process_ecg, 
    pre_process_pcg
)
from processing.transforms import (
    RandomStretch, 
    RandomTimeFreqMask
)
from util.fileio import read_ticking_PCG

class FeatureVectorsDataset_noWav(Dataset):
    def __init__(
        self, 
        data_dir,
        split_path,
        segment_dir,
        subset, 
        prepro_dir,
        fold=1, 
        sig_len=4.0, 
        channels=[1], 
        transform=None,
        fs=4125, 
        evaluate=False,
        **kwargs
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.split_path = os.path.abspath(split_path)
        self.segment_dir = os.path.abspath(segment_dir)
        self.fold = fold
        self.sig_len = sig_len
        self.fs = fs
        self.fs_new = fs
        self.subset = subset
        self.channels = channels
        self.channel = -1
        self.transform = transform
        self.evaluate = evaluate
        self.test_flag = 1 if subset == 'test' else 0
        self.train_flag = 1 if subset == 'train' else 0
        self.df = self._create_df() #this will now have cols 'frag':[list of channel frags], 'label':[int label]: and 'sub':[subject number]
        self.fragments = self.df['frag']
        self.labels = self.df['label']
        self.sub = self.df['sub']
        self.no_chan = len(self.fragments.iloc[0]) # this will access the first element and count entries

    def get_labels(self):
        """
        Get the labels of the dataset.
        
        Returns:
        - pd.Series: Series containing the labels of the dataset.
        """
        return self.labels

    def _create_df(self):
        """
        Create a DataFrame with columns 'frag', 'label', and 'sub'.
        This method should be overridden in subclasses to provide the actual data.
        """
        df_indicies = pd.read_csv(self.segment_dir)
        df_indicies["subject"] = df_indicies["subject"].astype(str)

        folds = pd.read_csv(self.split_path)
        split_str = f'split{self.fold}'
        data_split = folds[['patient','abnormality',split_str]]
        data_split_subset = data_split[data_split[split_str] == self.subset][['patient','abnormality']] 
        data_split_subset['ind'] = data_split_subset['patient'].map(lambda x: df_indicies.loc[df_indicies["subject"] == x.split("_")[0]].iloc[:, 1:].dropna(axis=1).astype(int).values.flatten().tolist() if x.split("_")[0] in df_indicies["subject"].values else [])

        df = FilenameLabelDFCreator(
            data_split_subset, 
            self.data_dir, 
            aug = 0
        ).segment_files_ind(
            no_seg=51, 
            seg_len=self.sig_len, 
            channels = self.channels, 
            fs_new = self.fs, 
            train_flag=1 if self.subset == 'train' else 0
        )

        return df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        channel_frag = self.fragments.iloc[idx] #This will have the multichannel (or single) fragments
        label = self.labels.iloc[idx]
        patient = self.sub.iloc[idx]
        
        if len(channel_frag)>1:
            # Create an empty array for multi-channel signals
            multi_sig = np.zeros((len(channel_frag[0]), self.no_chan))
            for i in range(len(channel_frag)):
                sig = channel_frag[i]
                multi_sig[:,i] = normalise_signal(sig)
        else:
            multi_sig = normalise_signal(channel_frag[0])

        if self.transform is not None:
            multi_sig = self.transform(multi_sig)

        label = f"{patient}.{1 if label == 1 else 0}" if self.evaluate else 1 if label == 1 else 0
        label = torch.tensor(label) if not self.evaluate else label
        data = torch.tensor(multi_sig, dtype=torch.float32)

        return {'input_vals' : data,
                'label': label}

    def augment(self, sig, fs):
        sig = RandomStretch(fs=fs).__call__(sig)
        sig = RandomTimeFreqMask(thickness=0.09, fs=fs).__call__(sig)
        sig = RandomTimeFreqMask(thickness=0.09 / 2, fs=fs).__call__(sig)
        return sig

class FilenameLabelDFCreator:
    def __init__(self, df, directory, aug: int):
        """
        Initialize the class with DataFrame and directory.

        Parameters:
        - df (pd.DataFrame): Original DataFrame with columns 'patient' and 'label'.
        - directory (str): Path to the directory containing patient recordings.
        """
        if 'patient' not in df.columns or 'abnormality' not in df.columns:
            raise ValueError("DataFrame must contain 'patient' and 'abnormality' columns")
        
        self.df = df
        self.directory = os.path.abspath(directory)
        self.aug = aug

    def create_filename_label_df(self): #THIS WILL NEVER BE USED _ I CONCATENATE ALL SIGNALS TOGETHER NOW
        """
        Create a DataFrame with filenames and corresponding labels based on patient identifiers.

        Returns:
        - pd.DataFrame: New DataFrame with columns 'filename' and 'label'.
        """
        new_data = []

        for patient in self.df['patient'].unique():
            label = self.df[self.df['patient'] == patient]['abnormality'].iloc[0]
            patient_files = [f for f in os.listdir(self.directory) if patient in f]
            if self.aug == 0:
                filtered_files = [f for f in patient_files if 'aug' not in f]
                patient_files = filtered_files
            
            for file in patient_files:
                new_data.append({'filename': os.path.join(self.directory,file), 'abnormality': label})

        return pd.DataFrame(new_data)

    def create_filename_label_df_one(self):
        """
        Create a DataFrame with filenames and corresponding labels based on patient identifiers,
        using only one 60s recording per patient.

        Returns:
        - pd.DataFrame: New DataFrame with columns 'filename' and 'label'.
        """
        new_data = []

        for patient in tqdm(self.df['patient'].unique(), desc="Creating filename-label DataFrame"):
            label = self.df[self.df['patient'] == patient]['abnormality'].iloc[0]
            ind = self.df[self.df['patient'] == patient]['ind'].iloc[0]
            patient_files = [f for f in os.listdir(self.directory) if patient in f]

            if len(patient_files) == 0:
                print(patient)
                input('error')
            if patient_files:
                # Choose only the first file found for this patient
                Mother_file = patient_files[0][0:16] #get the first 15 characters - unique recording

                patient_files_one = [g for g in os.listdir(self.directory) if Mother_file in g]
                if self.aug == 0:
                    filtered_files = [f for f in patient_files_one if 'aug' not in f]
                    patient_files_one = filtered_files
                    
                for file in patient_files_one:
                    new_data.append({'filename': os.path.join(self.directory,file), 'abnormality': label, 'ind': ind})
                #now get all of the fragments for this one 60s recording

        return pd.DataFrame(new_data)
    
    def segment_files(self, no_seg: int, seg_len: float, channels: list, fs_new: int, train_flag: int):
        #no_seg - number of segments to fragmeht the signal into
        #seg_len - length of fragment
        #train_flag - for balancing number of segments in training set only

        dataframe = self.create_filename_label_df_one()
        labels = dataframe['abnormality']
        # print(dataframe[0:5])
        if train_flag == 1:
            label0_no = len(labels) - sum(labels)
            label1_no = sum(labels)

            if label0_no <= label1_no : #this means there are more CAD
                no_seg0 = int(label1_no/label0_no * no_seg)
                no_seg1 = no_seg 
            else:
                no_seg1 = int(label0_no/label1_no * no_seg)
                no_seg0 = no_seg 
        else:
            no_seg0, no_seg1 = no_seg, no_seg #validation and test sets - same fragments for both
        fragments = [] #this is for the FeatureVectorDataSet eventually. append a list [np.array(frag), label, subject number]
        frag_len = int(seg_len*fs_new)

        for idx, row in dataframe.iterrows():
            # print(row['filename'])
            # input()
            # print(row['abnormality'])

            sub = row['filename'].split('/')[-1][0:5]
            # print(sub)
            # input()
            pcg_sig_all = [] #this will hold all of the relevent channels - still suitable for single channel data!!
            for c in channels:
                pcg_sig, fs = read_ticking_PCG(row['filename'], channel=c, noise_mic=0, collection=-1, max_len=60)
                pcg_sig = resample(pcg_sig, fs, fs_new) #this wont affect signla if already sampled at fs_new
                if c == 8:
                    pcg_sig = pre_process_ecg(pcg_sig, fs_new) #THIS IS ECG NOR PCG EVEN THO IT SAYS PCG_SIG
                else:
                    pcg_sig = pre_process_pcg(pcg_sig, fs_new)
                pcg_sig_all.append(pcg_sig)

            
            num_seg = no_seg0 if row['abnormality'] == 0 else no_seg1
            
            total_length = len(pcg_sig_all[0])
            required_coverage = frag_len * num_seg #seg and frag mean the same thing

            overlap = (required_coverage - total_length) // (num_seg-1)
            start = 0

            for i in range(num_seg-1):
                end = start + frag_len
                pcg_frag_all = []
                for idx, pcg in enumerate(pcg_sig_all): #grouping all the np.array channels into a list
                    pcg_frag_all.append(pcg[start:end])

                fragments.append([pcg_frag_all,row['abnormality'], sub])
                # start = end- overlap if i is not num_seg-2 else len(pcg_sig_all[0])-frag_len #give more overlap to last segment so it is same number of samples
                start = end - overlap if end-overlap+frag_len < len(pcg_sig_all[0]) else len(pcg_sig_all[0])-frag_len #give more overlap to last segment so it is same number of samples
                start = end - overlap  #give more overlap to last segment so it is same number of samples

        fragments_df = pd.DataFrame(fragments, columns = ["frag", "label", "sub"])
        return fragments_df
    
    #FIXME 
    #This is the function that will need the indicies incorporated in!!!
    def segment_files_ind(self, no_seg: int, seg_len: float, channels: list, fs_new: int, train_flag: int):
        #no_seg - number of segments to fragmeht the signal into
        #seg_len - length of fragment
        #train_flag - for balancing number of segments in training set only

        dataframe = self.create_filename_label_df_one()
        labels = dataframe['abnormality']
        labels_processed = [(label+1) // 2 for label in labels] if -1 in labels.to_list() else labels
        
        # print(dataframe[0:5])
        if train_flag == 1:
            label0_no = len(labels_processed) - sum(labels_processed)
            label1_no = sum(labels_processed)

            if label0_no <= label1_no : #this means there are more CAD
                no_seg0 = int(label1_no/label0_no * no_seg)
                no_seg1 = no_seg 
            else:
                no_seg1 = int(label0_no/label1_no * no_seg)
                no_seg0 = no_seg 
        else:
            no_seg0, no_seg1 = no_seg, no_seg #validation and test sets - same fragments for both
        fragments = [] #this is for the FeatureVectorDataSet eventually. append a list [np.array(frag), label, subject number]
        frag_len = int(seg_len*fs_new)

        for idx, row in tqdm(dataframe.iterrows(), desc="Segmenting files"):
            # print(row['filename'])
            # input()
            # print(row['abnormality'])

            sub = row['filename'].split('/')[-1][0:5]
            good_ind = row['ind']
            if len(good_ind) % 2 != 0:
                print('error - length of indicies must be even')
                input('ERROR')

            # print('')
            # print(sub)
            # print(good_ind)
            # input()
            keep = 1
            pcg_sig_all = [] #this will hold all of the relevent channels - still suitable for single channel data!!
            for c in channels:
                pcg_sig_concat, fs = read_ticking_PCG(row['filename'], channel=c, noise_mic=0, collection=-1, max_len=60) #This will be the concatenated signal
                pcg_sig_good = []
                for i in range(0, len(good_ind), 2): #THIS WILL EXTRACT THE GOOD AREAS OUT OF THE SIGNAL
                    # print(good_ind[i],good_ind[i+1])
                    pcg_sig_temp = pcg_sig_concat[good_ind[i]-1:good_ind[i+1]] #the indices are generated from matlab!!
                    # print(good_ind[i],good_ind[i+1],len(pcg_sig_temp))
                    # input('cunt')
                    # print(len(pcg_sig_temp))

                    if good_ind[i+1]-good_ind[i]>seg_len*fs:
                        # input('here')
                        pcg_sig_temp = resample(pcg_sig_temp,fs,fs_new) #this wont affect signla if already sampled at fs_new
                        pcg_sig_good.append(pcg_sig_temp)
                pcg_sig_all.append(pcg_sig_good)    
                if len(pcg_sig_good) == 0:
                    print('woah')
                    print('NO GOOD SIG FOR ',sub)
                    keep = 0
                 #THERE WILL BE MULTIPLE PARENT SEGMENTS PER CHANNEL NOW
            # print(channels)
            # print(len(pcg_sig_all))
            # print(len(pcg_sig_all[0]))
            if keep!=0:
                num_seg = no_seg0 if row['abnormality'] == 0 else no_seg1 #number of fragments to be extracted from the concatenated signal in total

                Parent_lengths = [len(pcg_sig_all[0][j]) for j in range(len(pcg_sig_all[0]))]
                # print(Parent_lengths)
                # print(sum(Parent_lengths))
                # input('HEY')

                #GET TOTAL NUMBER OF SEGMENTS TO EXTRACT FROM EACH PARENT SIGNAL
                exact_values = [Parent_lengths[j] / sum(Parent_lengths) * num_seg for j in range(len(Parent_lengths))] #store the amount of segments to extract from each Parent Segment
                Parent_segNum = [int(x) for x in exact_values]
                remaining = num_seg - sum(Parent_segNum)
                # Distribute the remaining segments to indices with the largest rounding errors
                errors = [(exact_values[j] - Parent_segNum[j], j) for j in range(len(Parent_lengths))]
                errors.sort(reverse=True, key=lambda x: x[0])  # Sort by largest rounding error
                # Assign remaining segments
                for i in range(remaining):
                    Parent_segNum[errors[i][1]] += 1

                # print(Parent_lengths)
                # print(Parent_segNum)

                for idx, num_seg in enumerate(Parent_segNum):
                    # xxx=1
                    pcg_sig_all_temp = [a[idx] for a in pcg_sig_all]
                    total_length = Parent_lengths[idx]
                    # print(num_seg, total_length, len(pcg_sig_all_temp))
                    required_coverage = frag_len * num_seg  # Total length needed to cover with fragments
                    step_size = (total_length - frag_len) // (num_seg - 1) if num_seg > 1 else 0  # Adjust for gaps if needed

                    start = 0

                    for i in range(num_seg):
                        end = start + frag_len
                        pcg_frag_all = [pcg[start:end] for pcg in pcg_sig_all_temp]  # Extracting fragments for all channels
                        # pcg_frag_all = [[1,9,idx] for pcg in pcg_sig_all_temp]  # for debugging

                        fragments.append([pcg_frag_all, row['abnormality'], sub])

                        # Adjust start position for next fragment
                        if num_seg > 1:
                            if required_coverage > total_length:  # Overlapping case
                                overlap = (required_coverage - total_length) // (num_seg-1)
                                start = end - overlap if end-overlap+frag_len < len(pcg_sig_all_temp[0]) else len(pcg_sig_all_temp[0])-frag_len #give more overlap to last segment so it is same number of samples
                            else:  # Gaps exist
                                start = min(start + step_size, len(pcg_sig_all_temp[0]) - frag_len)
                        # print(len(pcg_frag_all[0]), xxx)
                        # xxx+=1
                
                    # input('here')
        fragments_df = pd.DataFrame(fragments, columns = ["frag", "label", "sub"])
        return fragments_df

    def create_df(self, code):
        """
        Choose which function to use based on the provided code.

        Parameters:
        - code (str): Code indicating which function to use ('all' or 'one').

        Returns:
        - pd.DataFrame: Resulting DataFrame based on chosen method.
        """
        if code == 'all':
            return self.create_filename_label_df()
        elif code == 'one':
            return self.create_filename_label_df_one()
        else:
            raise ValueError("Invalid code. Use 'all' or 'one'.")
