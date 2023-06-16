Brain Computer-Interface Classification for Stroke Rehabilitation (DL)
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
### Python environment
To run the code, a conda environment is required.
Check the [Conda website](https://www.anaconda.com/) for how to install it.

### Installation
1. Clone the repo
   ```sh
   git clone git@github.com:AnnaGr-Git/BCI_stroke_rehab.git
   ```
2. Create and activate a new conda environment from .yaml file
   ```sh
   conda env create -f environment.yml
   conda activate BCI
   ```
3. Install further pre-requisites
   ```sh
   pip install -r requirements.txt
   ```
   or
   ```sh
   pip install -r requirements_all.txt
   ```
## Record EEG Data
### Hardware
* Open BCI EEG Headset with Dongle
* Computer with Open BCI UI (download: https://openbci.com/downloads)

### Measurement Setup
Follow the instructions in file documents/measurement_rules.txt

### Record Measurement
1. Open the Open-BCI UI and start the measurements
2. Use the python script "src/data/game_record_data.py"
   * Define user name with "USER = name"
   * Define the used channels in correct defined order with "CHANNELS = your channel names"
   * Define the sample rate with "SAMPLE_RATE = your sample rate" (OpenBCI samples with 125 Hz with 16 channels)

## Create BCI-Dataset
For own recordings, use the notebook "notebooks/CreatePhysioNet_Dataset.ipynb"; for PhysioNet data, use the notebook "notebooks/CreatePhysioNet_Dataset.ipynb"
1. A class structure with EEG-recordings (from csv or mat files) and corresponding information is created
2. Data can be easily validated and bandpass filtered
3. You can create CSP-features and plot them

## Create Training Set
After collecting the measurements in a dataset, a training set should be created before starting with the training of the model.
The notebook "notebooks/Create_Trainingset.ipynb" shows an example on how to do that.
You can choose between different options on how the label will be saved and how big the test size is.

## Train Model ML
To extract CSP-features from the data and classify them with different ML-methods, the notebook "notebooks/Train_Classifiers.ipynb" can be used.
It includes LDA, QDA, XGB, SVM, and a simple NN.

## Train Model DL
### Train on your own Computer (CPU)
* Adapt the selected training script from src/models: GPU = False
* Run the chosen training script in your favorite code editor 

### Train on DTU Cluster (GPU)
* Follow the instructions in the instruction file "documents/DTU_cluster_Instructions.pdf"
* Adapt the selected training script from src/models: GPU = True
* To select different training scripts for training, change the last line in the "submit.sh" file

## Evaluation
* Notebook "notebooks/Results_Evaluation.ipynb" visualizes the trained model performances in plots. These plots were used in the report.
* If you don't have the model performance files from training or you want to evaluate the model on your own:
  Select one of the following jupyter notebooks, depending on what to evaluate:
  (Change the model paths to the path of your trained model!)
    - Model Performance on selected test data: "notebooks/Evaluate_1D_CNN.ipynb"
    - Model Performance with different selected channel pairs: "notebooks/Evaluate_1D_CNN_channel_pairs.ipynb"
    - Model Performance with different selected test subjects: "notebooks/Evaluate_1D_CNN_test_subjects.ipynb"
    - Model Performance evaluated with 10-fold-Cross-Validation or Leave-One-Subject-Out: "notebooks/Evaluation.ipynb"

## Visualization
* The notebook "notebooks/Visualize.ipynb" shows some examples of how to visualize the recorded EEG data

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
