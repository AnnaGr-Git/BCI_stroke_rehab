# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne


from src.data.signal_processing import bandpass, csp, apply_mix, logvar, best_csp_components



class BCIDataset(Dataset):
    """
    A class to load the EEG-BCI dataset

    Attributes
    ----------
    data_folder : str
        path of directory where the recorded data is stored
    images : torch.tensor (num_samples x image_size[0] x image_size[1])
        images of the dataset, stored in one tensor
    labels : torch.tensor
        labels of each sample (num_samples x 1
        )

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self, data_root, subjects: List[str], measurements: List[str], classnames:List[str]=['arm_left', 'arm_right'], sample_rate:int=125, measurement_length:int=4, filetype:str="csv") -> None:
        self.data_root = data_root
        self.subjects = subjects
        self.measurements = measurements
        self.classes = classnames
        self.sample_rate = sample_rate
        self.measurement_length = measurement_length
        self.data_length = self.sample_rate * self.measurement_length
        if filetype == "mat":
            self.channel_names = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15','ch16']
            self.data = self.create_dataframe_mat()
            
        else:
            self.data = self.create_dataframe_csv()
            self.channel_names = list(self.data['sample'].iloc[0].columns[-16:])
        
        self.selected_data = "sample"
        self.csp_Matrix = np.array([])
        self.selected_csp_components = []
        self.training_data = {}

    def create_dataframe_csv(self):
        """Create dataframe of samples and their measurement information"""
        dataframe = {}
        print(f"Subjects: {self.subjects}")
        
        if len(self.measurements)<=0:
            for subj in self.subjects:
                # Get all measurements
                dirpath = os.path.join(self.data_root, subj+"/")
                if os.path.isdir(dirpath):
                    subfolders = [ os.path.basename(f.path) for f in os.scandir(dirpath) if f.is_dir()]
                    for sub in subfolders:
                        self.measurements.append(sub)
                else:
                    print("Dirpath is not a directory.")
                    continue

        for subj in self.subjects:
            for measurement in self.measurements:
                dirpath = os.path.join(self.data_root, subj+"/", measurement+"/")
                if os.path.isdir(dirpath):
                    directory = os.listdir(dirpath)
                else:
                    continue
                # Iterate through all measurement files
                for file in directory:
                    indexes = [ind for ind, ch in enumerate(file) if ch.lower() == '_']
                    
                    # Get data
                    filepath = self.data_root + subj+"/"+measurement+"/"+file
                    # shape: num_samples x num_channels
                    sample = pd.read_csv(filepath)

                    # Crop data to same length
                    if sample.shape[0] > (self.data_length - 1):
                        cropped_sample = sample.iloc[60:60+self.data_length, :]
                    else:
                        print("ERROR: Sample doesn't exceed minimum length.")
                        continue
                    
                    # Save information and data in dataframe
                    if len(dataframe)==0:
                        dataframe['subject'] = [subj]
                        dataframe['measurement'] = [measurement]
                        dataframe['class'] = [file[0:indexes[1]]]
                        dataframe['sampleID'] = [file[indexes[1]+1:indexes[2]]]
                        dataframe['path'] = [file]
                        dataframe['sample'] = [cropped_sample]
                    else:
                        dataframe['subject'].append(subj)
                        dataframe['measurement'].append(measurement)
                        dataframe['class'].append(file[0:indexes[1]])
                        dataframe['sampleID'].append(file[indexes[1]+1:indexes[2]])
                        dataframe['path'].append(file)
                        dataframe['sample'].append(cropped_sample)

        return pd.DataFrame(dataframe)
    
    def create_dataframe_mat(self):
        """Create dataframe of samples and their measurement information"""
        dataframe = {}
        if len(self.measurements)<=0:
            for subj in self.subjects:
                # Get all measurements
                dirpath = os.path.join(self.data_root, subj+"/")
                if os.path.isdir(dirpath):
                    subfolders = [ os.path.basename(f.path) for f in os.scandir(dirpath) if f.is_dir()]
                    for sub in subfolders:
                        self.measurements.append(sub)
                else:
                    continue

        for subj in self.subjects:
            for measurement in self.measurements:
                filepath = os.path.join(self.data_root, subj+"/", measurement+'.mat')
                if not os.path.isfile(filepath):
                    print(filepath)
                    print("File is not in directory.")
                    continue

                # Get data of measurement file
                mat_contents = scipy.io.loadmat(filepath)
                data = mat_contents['subjectData'][0][0]
                num_trials = np.shape(data['trialsLabels'])[0]
                num_datapoints, num_channels = np.shape(data['trialsData'][0][0])

                # Transform data in dataframe
                trial_data = data['trialsData']
                labels = data['trialsLabels'].flatten()

                for trial_idx in range(num_trials):
                    label = self.classes[labels[trial_idx]]
                    sample = trial_data[trial_idx][0]
                    sample_df = {}
                    sample_df['time_in_s'] = np.arange(0,self.measurement_length,self.measurement_length/num_datapoints)
                    # Get data of each channel
                    for ch in range(num_channels):
                        sample_df[self.channel_names[ch]] = sample[:,ch]
                        
                    sample_df = pd.DataFrame(sample_df)
                    
                    # Save information and data in dataframe
                    if len(dataframe)==0:
                        dataframe['subject'] = [subj]
                        dataframe['measurement'] = [measurement]
                        dataframe['class'] = [label]
                        dataframe['sampleID'] = [trial_idx]
                        dataframe['path'] = [filepath]
                        dataframe['sample'] = [sample_df]
                    else:
                        dataframe['subject'].append(subj)
                        dataframe['measurement'].append(measurement)
                        dataframe['class'].append(label)
                        dataframe['sampleID'].append(trial_idx)
                        dataframe['path'].append(filepath)
                        dataframe['sample'].append(sample_df)

        return pd.DataFrame(dataframe)

    
    def get_sample_values(self,idx, selected_data: str="sample")-> np.array:
        # sample shape: sample_points x channels
        channel_names = list(self.data['sample'].iloc[idx].columns[-16:])
        sample = self.data[selected_data].iloc[idx][channel_names]
        return np.array(sample)

    def get_data_array(self, selected_data: str="sample")-> np.array:
        sample_arr = np.array([])
        for s in range(len(self.data)):
            # Get sample as array
            sample = np.expand_dims(self.get_sample_values(s, selected_data), axis=0)
            if len(sample_arr) > 0:
                sample_arr = np.concatenate((sample_arr, sample), axis=0)
            else:
                sample_arr = sample
        return sample_arr


    def validate_data(self, volt:int=250):
        invalid_indexes = []
        for sample_idx in range(len(self.data)):
            sample = self.get_sample_values(sample_idx, selected_data="sample")
            max_R = np.array([max(sample[:,i]) for i in range(len(self.channel_names))])
            min_R = np.array([min(sample[:,i]) for i in range(len(self.channel_names))])
            if any(max_R > volt) or any(min_R < -volt):
                invalid_indexes.append(sample_idx)
            
        # delete invalid indexes from data
        print(f"Total invalid samples: {len(invalid_indexes)}")
        self.data=self.data.drop(self.data.index[invalid_indexes])

    def apply_bandpass_filtering(self, low_f: int=8, high_f:int=15, selected_data: str="sample")-> None:
        # required: array in shape (n_channels x num_datapoints x num_samples)
        data_arr = self.get_data_array(selected_data)
        data_arr = data_arr.transpose(2,1,0)

        data_filtered = bandpass(data_arr, low_f, high_f, self.sample_rate, len(self.channel_names), self.data_length)
        data_filtered = data_filtered.transpose(2,1,0)

        filtered_list = []
        for sample_idx in range(len(data_filtered)):
            filtered_sample = {}
            # Copy time from raw data
            columnames = self.data['sample'].iloc[sample_idx].columns
            if 'time_in_s' in columnames:
                filtered_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['time_in_s'])
            elif 's' in columnames:
                filtered_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['s'])
            else:
                print("Time data not found.")
                

            channel_names = list(columnames[-16:])
            for ch_idx in range(len(channel_names)):
                ch = channel_names[ch_idx]
                filtered_sample[ch] = list(data_filtered[sample_idx,:,ch_idx])
            filtered_list.append(pd.DataFrame(filtered_sample))

        self.data['filtered'] = filtered_list

    def calc_csp(self, only_train:bool=False, selected_data: str="filtered"):
        # Check if using only train samples for calculation
        if only_train:
            train_indexes = np.where(self.data["train_split"] == 'train')[0]
        else:
            train_indexes = np.array(range(len(self.data)))

        # Get samples of each class
        class_indexes = {}
        for cl in self.classes:
            class_indexes[cl] = list(np.where(self.data["class"] == cl)[0])

        # Choose indexes for training & selected class
        indexes_left = list(np.intersect1d(train_indexes, class_indexes[self.classes[0]]))
        indexes_right = list(np.intersect1d(train_indexes, class_indexes[self.classes[1]]))
        # print(f"Indexes left: {indexes_left}")
        # print(f"Indexes right: {indexes_right}")

        # Get data
        dataset_arr = self.get_data_array(selected_data)
        # Change shape from (trials x samples x channels) to (channels x samples x trials)
        dataset_arr = dataset_arr.transpose(2, 1, 0)

        # Calculate csp matrix
        W_mtx = csp(dataset_arr[:,:,indexes_right], dataset_arr[:,:,indexes_left], self.data_length)
        self.csp_Matrix = W_mtx
        return W_mtx

    def apply_csp(self, selected_data: str="filtered"):
        if len(self.csp_Matrix) > 0:
            # Get data
            dataset_arr = self.get_data_array(selected_data)
            # Change shape from (trials x samples x channels) to (channels x samples x trials)
            dataset_arr = dataset_arr.transpose(2, 1, 0)
            # Apply matrix to all data
            trials_csp = apply_mix(self.csp_Matrix, dataset_arr, len(self.channel_names), self.data_length).transpose(2, 1, 0)

            # Save csp in dataclass
            csp_list = []
            for sample_idx in range(len(trials_csp)):
                csp_sample = {}
                # Copy time from raw data
                columnames = self.data['sample'].iloc[sample_idx].columns
                if 'time_in_s' in columnames:
                    csp_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['time_in_s'])
                elif 's' in columnames:
                    csp_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['s'])
                else:
                    print("Time data not found.")

                channel_names = list(columnames[-16:])
                for ch_idx in range(len(channel_names)):
                    ch = channel_names[ch_idx]
                    csp_sample[ch] = list(trials_csp[sample_idx,:,ch_idx])
                csp_list.append(pd.DataFrame(csp_sample))

            self.data['csp'] = csp_list
        else:
            print("CSP-Matrix is not calculated yet. Please call function calc_csp first.")


    def feature_extraction_CSP(self, only_train:bool=False, selected_data:str='filtered', num_components:int=None):
        # Calc CSP Matrix
        csp_Matrix = self.calc_csp(only_train=only_train, selected_data=selected_data)
        # Apply CSP on all data
        self.apply_csp(selected_data=selected_data)

        # Check if using only train samples for calculation
        if only_train:
            train_indexes = np.where(self.data["train_split"] == 'train')[0]
        else:
            train_indexes = np.array(range(len(self.data)))

        # Get samples of each class
        class_indexes = {}
        for cl in self.classes:
            class_indexes[cl] = list(np.where(self.data["class"] == cl)[0])

        # Choose indexes for training & selected class
        indexes_left = list(np.intersect1d(train_indexes, class_indexes[self.classes[0]]))
        indexes_right = list(np.intersect1d(train_indexes, class_indexes[self.classes[1]]))
        # print(f"Indexes left: {indexes_left}")
        # print(f"Indexes right: {indexes_right}")

        # Get data as array
        data_arr = self.get_data_array(selected_data="csp")
        print(f"Shape data_array: {np.shape(data_arr)}")

        trials_csp = {self.classes[0]: data_arr[indexes_left,:], self.classes[1]: data_arr[indexes_right,:]}

        if num_components == None:
            self.selected_csp_components = list(range(np.shape(data_arr)[2]))
        else:
            components = best_csp_components(trials_csp, self.classes, num_comp=num_components)
            self.selected_csp_components = components

        print(f"Selected CSP components: {self.selected_csp_components}")

        csp_data = data_arr[:,:,self.selected_csp_components]
        print(f"Shape CSP data: {np.shape(csp_data)}")
        return csp_data

    def create_random_train_test_split(self, test_size):
        # Split data in train and test samples
        sample_indexes = list(range(len(self.data)))
        class_labels = list(self.data['class'])

        indexes_train, indexes_test, y_train, y_test = train_test_split(sample_indexes, class_labels,
                                                            stratify=class_labels, 
                                                            test_size=test_size,
                                                            random_state=54)    # 42
        print(f"Split dataset in {len(indexes_train)} train and {len(indexes_test)} test samples.")
        #print(f"Test indexes: {indexes_test}")
        
        train_test_list = []
        for i in range(len(self.data)):
            if i in indexes_train:
                train_test_list.append("train")
            elif i in indexes_test:
                train_test_list.append("test")
            else:
                print("Index is not used in training.")
                
        self.data['train_split'] = train_test_list

        return indexes_train, indexes_test, y_train, y_test
    
    def create_subjects_test_split(self, test_subjects:list):
        sample_indexes = list(range(len(self.data)))

        indexes_test = []
        for subj in test_subjects:
            subj_indexes = np.where(self.data["subject"] == subj)[0]
            indexes_test.append(subj_indexes)
            
        indexes_test = list(np.array(indexes_test).flatten())
        indexes_train = list(set(sample_indexes) ^ set(indexes_test))
        print(f"Split dataset in {len(indexes_train)} train and {len(indexes_test)} test samples.")

        train_test_list = []
        for i in range(len(self.data)):
            if i in indexes_train:
                train_test_list.append("train")
            elif i in indexes_test:
                train_test_list.append("test")
            else:
                print("Index is not used in training.")
                
        self.data['train_split'] = train_test_list

        class_labels = np.array(self.data['class'])
        y_train = class_labels[indexes_train]
        y_test = class_labels[indexes_test]

        return indexes_train, indexes_test, y_train, y_test
        

    def create_training_data(self, test_size:float=0.2, mode: str = ["class_as_array","class_as_key"][0], selected_data:str='filtered', num_components:int=None, test_subjects:list=[]):
        # Split data in train and test samples
        if len(test_subjects)==0:
            print("Split train/test-data randomly.")
            indexes_train, indexes_test, y_train, y_test = self.create_random_train_test_split(test_size)
        else:
            print(f"Split data using {test_subjects} for testing.")
            indexes_train, indexes_test, y_train, y_test = self.create_subjects_test_split(test_subjects)

        # Feature Extraction
        data_2comp = self.feature_extraction_CSP(only_train=True,selected_data=selected_data, num_components=num_components)
        # Calc logvar of features
        features = logvar(data_2comp)

        # Create training data structure
        if mode == "class_as_key":
            # Get samples of each class
            class_indexes = {self.classes[0]:[], self.classes[1]:[]}
            for cl in self.classes:
                class_indexes[cl] = list(np.where(self.data["class"] == cl)[0])
                
            train_indexes_class0 = list(np.intersect1d(indexes_train, class_indexes[self.classes[0]]))
            train_indexes_class1 = list(np.intersect1d(indexes_train, class_indexes[self.classes[1]]))
            test_indexes_class0 = list(np.intersect1d(indexes_test, class_indexes[self.classes[0]]))
            test_indexes_class1 = list(np.intersect1d(indexes_test, class_indexes[self.classes[1]]))

            training_data = {self.classes[0]: features[train_indexes_class0], self.classes[1]: features[train_indexes_class1]}
            test_data = {self.classes[0]: features[test_indexes_class0], self.classes[1]: features[test_indexes_class1]}
             
        else:
            # Get trainset and testset
            X_train = features[indexes_train]
            X_test = features[indexes_test]

            print(f"Shape X_train: {np.shape(X_train)}")

            # Transform y from classnames into int
            classes = np.unique(y_train)
            y_train_int = []
            for y in y_train:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_train_int.append(cl_idx)
            y_test_int = []
            for y in y_test:
                for cl_idx in range(len(classes)):
                    if y == classes[cl_idx]:
                        y_test_int.append(cl_idx)
            print(f"Value of label corresponds to position in class array: {classes}")

            training_data = {"X":X_train, "y":np.array(y_train_int)}
            test_data = {"X":X_test, "y":np.array(y_test_int)}
        
        self.training_data = {"train":training_data, "test":test_data}
        return self.training_data
    
    def load_subject_data(self, sub, channels, leave_subj_out:bool=False):
        if leave_subj_out:
            # Get data of all subjects except the selected one
            sub_df = self.data.loc[self.data['subject'] != sub]
            print(f"Get data from subjects: {np.unique(sub_df['subject'])}")
        else:
            # Get data of selected subject
            sub_df = self.data.loc[self.data['subject'] == sub]
            print(f"Get data from subjects: {np.unique(sub_df['subject'])}")

        # Get data
        data = np.array([])
        labels = []
        ch_pairs = []
        for ch_pair in channels:
            #print(f"Channel Pair: {ch_pair}")
            for trial in range(len(sub_df)):
                labels.append(sub_df.iloc[trial]["class"])
                ch_pairs.append(ch_pair)
                arr = np.expand_dims(np.swapaxes(np.array(sub_df.iloc[trial]["filtered"][ch_pair]),0,1), axis=0)
                if len(data) > 0:
                    data = np.concatenate((data, arr), axis=0)
                else:
                    data = arr
            #print(f"Data shape: {np.shape(data)}")

        label_translation = {"arm_left":"L", "arm_right":"R"}
        labels_transl = np.array([label_translation[l] for l in labels])

        return data, labels_transl, np.array(ch_pairs)

    def get_shapes(self):
        shapes = {}
        shapes['n_channels']= len(self.channel_names)
        shapes['sample_length'] = self.data['sample'].iloc[0].shape[0]

        shapes['n_classes'] = {}
        for cl in self.classes:
            value_counts = self.data['class'].value_counts()
            if cl in list(value_counts.keys()):
                shapes['n_classes'][cl] = value_counts[cl]
            else:
                shapes['n_classes'][cl] = 0
        
        return shapes

    
    def __len__(self) -> int:
        """Return len of dataset"""
        return self.data.shape[0]
    
    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
    def __getitem__(self, idx: int):
        """Get next item in the dataset"""
        sample = self.data['sample'].loc[idx]
        label = self.data['class'].loc[idx]
        return (sample, label)
    



"""
Code used from:
https://github.com/ambitious-octopus/MI-EEG-1D-CNN 

A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro
"""

class PhysionetDataset(Dataset):
    def __init__(self ,data_root, subjects, runs, target_sample_rate=None, measurement_length:float=4.0, tmin:float=0.0):
        self.data_root = data_root
        self.subjects = subjects
        self.runs = runs
        self.classes = []
        self.sample_rate = 160
        self.target_sample_rate = target_sample_rate
        self.measurement_length = measurement_length
        self.data_length = self.sample_rate * self.measurement_length
        self.tmin = tmin
        self.tmax = self.tmin + self.measurement_length
        self.channel_names = ["Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "Fz", "Fp1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3"]
        self.channel_sets = []
        self.exclude_base:bool
        self.class_4_flag = False

    def load_data(self, subject):
        all_subject_list = []
        sub = str(subject)
        runs = [str(r) for r in self.runs]

        # Tasks with imagery movement l/r
        task2 = [4, 8, 12]
        # Tasks with imagery movement both hands/both feet
        task4 = [6, 10, 14]

        if len(sub) == 1:
            sub_name = "S"+"00"+sub
        elif len(sub) == 2:
            sub_name = "S"+"0"+sub
        else:
            sub_name = "S"+sub
        sub_folder = os.path.join(self.data_root, sub_name)
        single_subject_run = []
        for run in runs:
            if len(run) == 1:
                path_run = os.path.join(sub_folder, sub_name+"R"+"0"+run+".edf")
            else:
                path_run = os.path.join(sub_folder, sub_name+"R"+ run +".edf")
            raw_run = read_raw_edf(path_run, preload=True)
            len_run = np.sum(raw_run._annotations.duration)
            if len_run > 124:
                print(sub)
                raw_run.crop(tmax=124)

            """
                B indicates baseline
                L indicates motor imagination of opening and closing left fist;
                R indicates motor imagination of opening and closing right fist;
                LR indicates motor imagination of opening and closing both fists;
                F indicates motor imagination of opening and closing both feet.
            """
            if int(run) in task2:
                for index, an in enumerate(raw_run.annotations.description):
                    if an == "T0":
                        raw_run.annotations.description[index] = "B"
                    if an == "T1":
                        raw_run.annotations.description[index] = "L"
                    if an == "T2":
                        raw_run.annotations.description[index] = "R"
            if int(run) in task4:
                    self.class_4_flag = True
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "LR"
                        if an == "T2":
                            raw_run.annotations.description[index] = "F"
            single_subject_run.append(raw_run)
        all_subject_list.append(single_subject_run)
        return all_subject_list

    def process_data(self, list_runs, channels):
        # Concatenate runs
        raw_conc_list = []
        for subj in list_runs:
            raw_conc = concatenate_raws(subj)
            raw_conc_list.append(raw_conc)

        # Delete "BAD boundary" and "EDGE boundary" from raws
        list_raw = []
        for subj in raw_conc_list:
            indexes = []
            for index, value in enumerate(subj.annotations.description):
                if value == "BAD boundary" or value == "EDGE boundary":
                    indexes.append(index)
            subj.annotations.delete(indexes)
            list_raw.append(subj)

        # Standardize montage of the raws
        raw_setted = []
        for subj in list_raw:
            eegbci.standardize(subj)
            montage = make_standard_montage('standard_1005')
            subj.set_montage(montage)
            raw_setted.append(subj)

        # Slice channels
        s_list = []
        for raw in raw_setted:
            s_list.append(raw.pick_channels(channels))

        # Split the original BaseRaw into numpy epochs
        """ 
        B indicates baseline
        L indicates motor imagination of opening and closing left fist;
        R indicates motor imagination of opening and closing right fist;
        """
            
        xs = list()
        ys = list()
        for raw in s_list:
            if self.exclude_base:
                if self.class_4_flag:
                    event_id = dict(F=2, L=3, LR=4, R=5)
                else:
                    event_id = dict(L=1, R=2)
            else:
                if self.class_4_flag:
                    event_id = dict(B=1, F=2, L=3, LR=4, R=5)
                else:
                    event_id = dict(B=0, L=1, R=2)

            events, _ = mne.events_from_annotations(raw, event_id=event_id)

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                exclude='bads')
            epochs = Epochs(raw, events, event_id, self.tmin, self.tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
            
            ## Resample data
            if self.target_sample_rate is not None:
                epochs = epochs.copy().resample(sfreq=self.target_sample_rate)

            y = list()
            for index, data in enumerate(epochs):
                y.append(epochs[index]._name)

            xs.append(np.array([epoch for epoch in epochs]))
            ys.append(y)

        x, y = np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]

        return x, y


    def generate_dataset(self, base_path, channel_sets, exclude_baseline:bool=True):
        self.channel_sets = channel_sets
        self.exclude_base = exclude_baseline

        # Iterate over channel pairs
        for channel_couple in self.channel_sets:
            save_path = os.path.join(base_path, channel_couple[0] + channel_couple[1])
            os.mkdir(save_path)
            # Iterate over subjects
            for sub in self.subjects:
                print(f"Subject: {sub}")
                x, y = self.process_data(self.load_data(subject=sub), channel_couple)

                np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
                np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)










@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
