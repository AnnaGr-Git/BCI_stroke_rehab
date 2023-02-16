# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import torch

from torch.utils.data import Dataset


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
        self.classes = {'hand_right': 0, 'hand_left': 1, 'wrist_right': 2, 'wrist_left': 3}
        self.sample_rate = sample_rate
        self.measurement_length = measurement_length
        self.data = self.create_dataframe()

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
                    sample = pd.read_csv(os.path.abspath(filepath))

                    # Crop data to same length
                    data_length = self.sample_rate * self.measurement_length
                    # if sample.shape[1] > (data_length - 1):
                    #     final_data[action].append(data[:, 125:data_length])
                    
                    # Save information and data in dataframe
                    if len(dataframe)==0:
                        dataframe['subject'] = [subj]
                        dataframe['measurement'] = [measurement]
                        dataframe['class'] = [file[0:indexes[1]]]
                        dataframe['sampleID'] = [file[indexes[1]+1:indexes[2]]]
                        dataframe['path'] = [file]
                        dataframe['data'] = [sample]
                    else:
                        dataframe['subject'].append(subj)
                        dataframe['measurement'].append(measurement)
                        dataframe['class'].append(file[0:indexes[1]])
                        dataframe['sampleID'].append(file[indexes[1]+1:indexes[2]])
                        dataframe['path'].append(file)
                        dataframe['data'].append(sample)

        return pd.DataFrame(dataframe)

    def validate_data(self):
        print("")
    
    def __len__(self) -> int:
        """Return len of dataset"""
        return self.data.shape[0]
    
    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
    def __getitem__(self, idx: int):
        """Get next item in the dataset"""
        sample = self.data['data'].loc[idx]
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
