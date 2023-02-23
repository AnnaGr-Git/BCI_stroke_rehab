# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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

    def __init__(self, data_root, subjects: List[str], measurements: List[str], sample_rate:int=125, measurement_length:int=4) -> None:
        self.data_root = data_root
        self.subjects = subjects
        self.measurements = measurements
        self.classes = ['arm_left', 'arm_right']
        self.sample_rate = sample_rate
        self.measurement_length = measurement_length
        self.data_length = self.sample_rate * self.measurement_length
        self.data = self.create_dataframe()
        self.channel_names = list(self.data['sample'].iloc[0].columns[-16:])
        self.selected_data = "sample"
        self.csp_Matrix = np.array([])
        self.selected_csp_components = []
        self.training_data = {}

    def create_dataframe(self):
        """Create dataframe of samples and their measurement information"""
        dataframe = {}
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
                    sample = pd.read_csv(os.path.abspath(filepath)) 

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
    
    def get_sample_values(self,idx, selected_data: str="sample")-> np.array:
        # sample shape: sample_points x channels
        sample = self.data[selected_data].iloc[idx][self.channel_names]
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
            filtered_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['time_in_s'])
            for ch_idx in range(len(self.channel_names)):
                ch = self.channel_names[ch_idx]
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
        class_indexes = {'arm_left':[], 'arm_right':[]}
        for cl in ['arm_left', 'arm_right']:
            class_indexes[cl] = list(np.where(self.data["class"] == cl)[0])

        # Choose indexes for training & selected class
        indexes_left = list(np.intersect1d(train_indexes, class_indexes['arm_left']))
        indexes_right = list(np.intersect1d(train_indexes, class_indexes['arm_right']))
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
                csp_sample['time_in_s'] = list(self.data['sample'].iloc[sample_idx]['time_in_s'])
                for ch_idx in range(len(self.channel_names)):
                    ch = self.channel_names[ch_idx]
                    csp_sample[ch] = list(trials_csp[sample_idx,:,ch_idx])
                csp_list.append(pd.DataFrame(csp_sample))

            self.data['csp'] = csp_list
        else:
            print("CSP-Matrix is not calculated yet. Please call function calc_csp first.")


    def feature_extraction_CSP(self, only_train:bool=False):
        # Calc CSP Matrix
        csp_Matrix = self.calc_csp(only_train=only_train, selected_data='filtered')
        # Apply CSP on all data
        self.apply_csp(selected_data="filtered")

        # Check if using only train samples for calculation
        if only_train:
            train_indexes = np.where(self.data["train_split"] == 'train')[0]
        else:
            train_indexes = np.array(range(len(self.data)))

        # Get samples of each class
        class_indexes = {self.classes[0]:[], self.classes[1]:[]}
        for cl in self.classes:
            class_indexes[cl] = list(np.where(self.data["class"] == cl)[0])

        # Choose indexes for training & selected class
        indexes_left = list(np.intersect1d(train_indexes, class_indexes['arm_left']))
        indexes_right = list(np.intersect1d(train_indexes, class_indexes['arm_right']))
        # print(f"Indexes left: {indexes_left}")
        # print(f"Indexes right: {indexes_right}")

        # Get data as array
        data_arr = self.get_data_array(selected_data="csp")
        trials_csp = {self.classes[0]: data_arr[indexes_left,:], self.classes[1]: data_arr[indexes_right,:]}

        components = best_csp_components(trials_csp, self.classes)
        self.selected_csp_components = components
        print(f"Best CSP components: {components}")

        return data_arr[:,:,components]

    def create_train_test_split(self, test_size):
        # Split data in train and test samples
        sample_indexes = list(range(len(self.data)))
        class_labels = list(self.data['class'])

        indexes_train, indexes_test, y_train, y_test = train_test_split(sample_indexes, class_labels,
                                                            stratify=class_labels, 
                                                            test_size=test_size)
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

        return indexes_train, indexes_test, y_train, y_test

    def create_training_data(self, test_size:float=0.2, mode: str = ["class_as_array","class_as_key"][0]):
        # Split data in train and test samples
        indexes_train, indexes_test, y_train, y_test = self.create_train_test_split(test_size)

        # Feature Extraction
        data_2comp = self.feature_extraction_CSP(only_train=True)
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
            
            training_data = {"X":X_train, "y":y_train}
            test_data = {"X":X_test, "y":y_test}
        
        self.training_data = {"train":training_data, "test":test_data}
        return self.training_data
        






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
