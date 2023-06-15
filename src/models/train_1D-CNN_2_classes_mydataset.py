"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
#sys.path.append("/workspace")
from src.models.model_architectures.model_1DCNN import HopefullNet
import numpy as np
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from src.data.general_processor import Utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.preprocessing import minmax_scale
import pathlib
from datetime import datetime

from src.data.make_dataset import BCIDataset

CHANNEL_PAIRS = "ours_3_pairs"
# CHANNEL_PAIRS = "ours_6_pairs"
GPU = False


root_path = pathlib.Path().resolve()
date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

# DATA
data_root = os.path.join(root_path,"data/raw/")
print(f"Data_root={data_root}")

# MODEL
save_path = os.path.join(root_path, "models/1D_CNN/" + CHANNEL_PAIRS + "/leave_subjects_out/")

try:
    os.mkdir(save_path)
except OSError as error:
    print(error)  

## Device settings
if GPU:
    # GPU setting
    tf.autograph.set_verbosity(0)
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    # CPU setting
    tf.autograph.set_verbosity(0)
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print(physical_devices)



# Params
if CHANNEL_PAIRS == "ours_3_pairs":
    channels = [["C3","C4"],["F3","F4"],["P3","P4"]]
elif CHANNEL_PAIRS == "ours_6_pairs":
    channels = [["C3","C4"],["F3","F4"],["P3","P4"],["FP1","FP2"],["F7","F8"],["T3","T4"]]
else:
    print("Please select a valid channel pair setting!")
    channels = []

# Test subjects for testing transfer learning
subjects = ["dani", "ivo", "pablo", "huiyu", "manu", "fabio", "anna", "luisa", "sarah", "irene", "jan"]
measurements = []

# Get dataset
trainingset = BCIDataset(data_root, subjects, [], measurement_length=4)
trainingset.validate_data()
trainingset.apply_bandpass_filtering(selected_data="sample")


for sub in subjects:
    print(f"Train model for test subject {sub}")

    #Load data
    x, y, _ = trainingset.load_subject_data(sub, channels, leave_subj_out=True)

    #Reshape for scaling
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    #Grab a test set before SMOTE
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                                    y,
                                                                                    stratify=y,
                                                                                    test_size=0.20,
                                                                                    random_state=42)


    #Scale indipendently train/test
    x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
    x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

    #Create Validation/test
    x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                        y_valid_test_raw,
                                                        stratify=y_valid_test_raw,
                                                        test_size=0.50,
                                                        random_state=42)

    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], x.shape[1], x.shape[2]).astype(np.float64)
    x_valid = np.swapaxes(x_valid,1,2)
    x_test = x_test_raw.reshape(x_test_raw.shape[0], x.shape[1], x.shape[2]).astype(np.float64)
    x_test = np.swapaxes(x_test,1,2)


    ## Data augmentation with SMOTE
    # Resample to specific number
    if CHANNEL_PAIRS == "ours_3_pairs":
        sampling_class_counts = {'L': 6000, 'R': 6000}
    elif CHANNEL_PAIRS == "ours_6_pairs":
        # Before: ~5500 --> after: 22000
        sampling_class_counts = {'L': 11000, 'R': 11000}
    else:
        num = int(len(y_train_raw)*4)
        sampling_class_counts = {'L': num, 'R': num}


    sm = SMOTE(sampling_strategy=sampling_class_counts, random_state=42)

    x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)

    print('Classes Count:')
    values, counts = np.unique(y_train_raw, return_counts=True)
    print (f"Before oversampling: values={values}, counts={counts}")
    values, counts = np.unique(y_train, return_counts=True)
    print (f"After oversampling: values={values}, counts={counts}")

    x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], x.shape[1], x.shape[2]).astype(np.float64)
    x_train = np.swapaxes(x_train,1,2)

    # Transform labels into 0/1 values
    y_train_01 = []
    for y_label in y_train:
        if y_label == 'L':
            y_train_01.append(0)
        elif y_label == 'R':
            y_train_01.append(1)
        else:
            print("Train Labels are different than L or R...")
    y_valid_01 = []
    for y_label in y_valid:
        if y_label == 'L':
            y_valid_01.append(0)
        elif y_label == 'R':
            y_valid_01.append(1)
        else:
            print("Valid Labels are different than L or R...")
    y_test_01 = []
    for y_label in y_test:
        if y_label == 'L':
            y_test_01.append(0)
        elif y_label == 'R':
            y_test_01.append(1)
        else:
            print("Test Labels are different than L or R...")

    y_train = np.array(y_train_01)
    y_valid = np.array(y_valid_01)
    y_test = np.array(y_test_01)


    ## Model training
    learning_rate = 1e-4

    # loss = tf.keras.losses.categorical_crossentropy
    loss = tf.keras.losses.binary_crossentropy

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print(np.shape(x_train)[1])
    print(np.shape(x_train)[2])

    model = HopefullNet(inp_shape = (np.shape(x_train)[1],np.shape(x_train)[2]), two_class=True) # 640,2


    modelFolder = save_path+"test_"+sub+"/correctedBug/"
    try:
        os.mkdir(modelFolder)
    except OSError as error:
        print(error)

    modelPath = os.path.join(modelFolder, 'bestModel.h5')
    finalModelPath = os.path.join(modelFolder, 'finalModel.h5')

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint( # set model saving checkpoints
        modelPath, # set path to save model weights
        monitor='val_loss', # set monitor metrics
        verbose=1, # set training verbosity
        save_best_only=True, # set if want to save only best weights
        save_weights_only=True, # set if you want to save only model weights
        mode='auto', # set if save min or max in metrics
        save_freq="epoch" # interval between checkpoints
        )

    earlystopping = EarlyStopping(
        monitor='val_loss', # set monitor metrics
        min_delta=0.001, # set minimum metrics delta
        patience=4, # number of epochs to stop training
        restore_best_weights=True, # set if use best weights or last weights
        )
    callbacksList = [checkpoint, earlystopping] # build callbacks list


    print(f"Shape x_valid: {np.shape(x_valid)}")
    print(f"Shape y_valid: {np.shape(y_valid)}")


    hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                    validation_data=(x_valid, y_valid), callbacks=callbacksList) #32

    with open(os.path.join(modelFolder, "hist.pkl"), "wb") as file:
        pickle.dump(hist.history, file)

    model.save_weights(finalModelPath)

    """
    Test model
    """

    del model # Delete the original model, just to be sure!

     ## LOAD MODEL
    input_shape = (None, np.shape(x_train)[1], np.shape(x_train)[2])
        
    model = HopefullNet(inp_shape = (input_shape[1], input_shape[2]), two_class=True) # 500,2
    model.build(input_shape)
    model.load_weights(finalModelPath)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    testLoss, testAcc = model.evaluate(x_test, y_test)
    print('\nAccuracy:', testAcc)
    print('\nLoss: ', testLoss)

    from sklearn.metrics import classification_report, confusion_matrix
    yPred = model.predict(x_test)

    # convert from one hot encode in string
    yTestClass = y_test
    yPredClass = []
    for label in yPred:
        if label<0.5:
            yPredClass.append(0)
        elif label>=0.5:
            yPredClass.append(1)
        else:
            print("Label not found.")

    classif_report = classification_report(
                                yTestClass,
                                yPredClass,
                                target_names=["L", "R"]
                                )
    print('\n Classification report \n\n', classif_report)
    f = open(os.path.join(save_path,"classification_report_"+date+".pkl"),"wb")
    pickle.dump(classif_report,f)
    f.close()

    tn, fp, fn, tp = confusion_matrix(
                    yTestClass,
                    yPredClass,
                    ).ravel()
    conf_matrix = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    print('\n Confusion matrix \n\n', conf_matrix)
    f = open(os.path.join(save_path,"confusion_matrix_"+date+".pkl"),"wb")
    pickle.dump(conf_matrix,f)
    f.close()
