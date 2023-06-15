DL Classification for BCI Stroke Rehabilitation
==============================
Brain Computer-Interfaces (BCI) proved to be an effective method for the rehabilitation of post-stroke impairments by translating brain signals into movement intentions and supporting the execution of the motion with external devices such as exoskeletons. This is one of many applications through which the classification of electroencephalography (EEG) recordings of motor imagery (MI) tasks gained a lot of attention in current research.

This work presents an approach to how DL models can be trained to classify EEG-MI recordings with only a few data available. By making
use of the public dataset PhysioNet, a base model is trained with data from many subjects, learning to extract features from EEG data. This pre-learned knowledge is exploited when fine-tuning the model on one specific subject with little data from our recorded dataset. After fine-tuning the PhysioNet base model on 11 individual subjects and selecting the most robust channel pairs, a median accuracy of 98% is reached. Overall, the approach of fine-tuning with a different dataset offers new insights for the development of individualized
real-world BCI applications where limited data is available.

## Project Overview
![Alt text](reports/images/system_overview_low_quality.png?raw=true "Flowchart")

## Project Organization

    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The processed data for usage in this project
    │   └── raw            <- The original recorded data
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── documents          <- Other documents for explanation
    │
    ├── models             <- Trained and fine-tuned models, training and performance summaries
    │
    ├── notebooks          <- Jupyter notebooks as an interface to run the python scripts
    │
    ├── reports            <- Generated analysis and explanation files
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── images         <- Images to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment with necessary libraries,
    │                         generated with `pipreqs --encoding=utf8`
    ├── requirements_all.txt   <- The requirements file of all pip packages and their versions for reproducing the analysis environment,
    │                         generated with `pip freeze > requirements.txt`
    ├── environment.yml    <- The environment file to recreate the anaconda environment to run the project
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── game_record_data.py  <- Record EEG-measurements with python interface (MI-protocol)
    │   │   └── make_dataset.py      <- Functions and class structure to create an organized dataset from recordings
    │   │   └── signal_processing.py <- Signal processing functions
    │   │   └── general_processor.py <- Data processing functions for PhysioNet database
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   │                 
    │   │   ├── model_architectures
    │   │       └── model_1DCNN.py           <- Model architecture
    │   │   ├── ML_models.py                 <- Collection of ML methods
    │   │   └── train_1D-CNN.py              <- Train model with PhysioNet data (Replicate 1DCNN paper)
    │   │   └── train_1D-CNN_2_classes.py    <- Train model with PhysioNet data, only predicting 2 classes
    │   │   └── train_1D-CNN_2_classes_mydataset.py  <- Train model with own data, predicting 2 classes
    │   │   └── subject_specific_1D-CNN_mydataset.py  <- Train model for each subject of own data
    │   │   └── finetune_1D-CNN_mydataset.py  <- Fine-tune model on own subjects
    │   │   └── finetune_1D-CNN_physionet     <- Fine-tune model on subjects of PhysioNet
    │   │   └── validate_models.py  <- Methods to validate the model performance
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │       └── plotting.py
    │
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── submit.sh          <- File to run the code in DTU cluster


## Getting Started

## Train Model

## Evaluation

## Visualization

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
