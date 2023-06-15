import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly

from src.data.make_dataset import BCIDataset, PhysionetDataset

#Params
data_root = "C:/Users/annag/OneDrive - Danmarks Tekniske Universitet/Semester_04/Special_Course_BCI/03_code/BCI_stroke_rehab/data/raw/"

#channels = [["C3","C4"],["F3","F4"],["P3","P4"]]
channels = [["CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "FZ", "FP1", "FP2", "F7", "F3", "F4", "F8", "T3", "C3"]]

# Test subjects for testing transfer learning
test_subjects = ["dani", "ivo", "pablo", "huiyu", "manu", "fabio", "anna", "luisa", "sarah", "irene", "jan"]
measurements = []

# Get dataset
trainingset = BCIDataset(data_root, test_subjects, [], measurement_length=4)
trainingset.validate_data()
trainingset.apply_bandpass_filtering(selected_data="sample")

# Select subject for visualization
sub = "dani"

#Load data
x, y = trainingset.load_subject_data(sub, channels)
#x, y = Utils.load(channels, [sub], base_path=source_path)

#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

#Grab a test set before SMOTE
x_train_raw, x_test_raw, y_train_raw, y_test = train_test_split(reshaped_x,
                                                                y,
                                                                stratify=y,
                                                                test_size=0.20,
                                                                random_state=42)

#Scale indipendently train/test
#Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_scaled_raw = minmax_scale(x_test_raw, axis=1)

x_test = x_test_scaled_raw.reshape(x_test_scaled_raw.shape[0], int(x_test_scaled_raw.shape[1]/x.shape[1]),x.shape[1]).astype(np.float64)
x_train = x_train_scaled_raw.reshape(x_train_scaled_raw.shape[0], int(x_train_scaled_raw.shape[1]/x.shape[1]), x.shape[1]).astype(np.float64)

print(np.shape(x))
print(np.shape(x_train))
print(np.shape(x_test))

## Create MNE INFO
sampling_freq = trainingset.sample_rate
ch_names = ["Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "Fz", "Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3"]
ch_types = ['eeg'] * 16
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')
info['description'] = 'My recorded dataset'
print(info)

## Create MNE RAW data
# Transform shape from [num_trials x n_samples x num_channels into [num_trials x num_channels x n_samples]
data = np.transpose(x_train, (0, 2, 1))
print(np.shape(data))
# data = [num_channels x n_samples]
data_trial = data[0]
print(np.shape(data_trial))
trial_raw = mne.io.RawArray(data_trial, info)

## Plot data
start, stop = trial_raw.time_as_index([0.5, 3.5])  # 0.5 s to 3.5 s data segment
data, times = trial_raw[:, start:stop]

plt.plot(times, data.T)
plt.xlabel('time (s)')
plt.ylabel('MEG data (T)')

update = dict(layout=dict(showlegend=True), data=[dict(name=trial_raw.info['ch_names'][ch]) for ch in range(len(ch_names))])
py.iplot_mpl(plt.gcf(), update=update)

plt.show()