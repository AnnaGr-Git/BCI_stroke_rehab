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
from src.models.model_architectures.model_1DCNN import HopefullNet
import numpy as np
from datetime import datetime
import pathlib
import tensorflow as tf
from src.data.general_processor import Utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.preprocessing import minmax_scale
from collections import Counter

from src.data.make_dataset import BCIDataset

GPU = True

BASE_MODEL = "physionet"
# BASE_MODEL = "mydataset"

# CHANNEL_PAIRS = "ours_3_pairs"
CHANNEL_PAIRS = "ours_6_pairs"


root_path = pathlib.Path().resolve()
date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

# DATA
data_root = os.path.join(root_path,"data/raw/")
model_directory = "models/1D_CNN/"+CHANNEL_PAIRS+"/leave_subjects_out/"

if BASE_MODEL == "physionet":
    # SOURCE MODEL
    if CHANNEL_PAIRS == "ours_3_pairs":
        print("Take Physionet Basemodel trained on 3 channel pairs.")
        model_folder = "test_physio_34_10_65_90_101/2023-04-25_16-09-09_correctedBug/"
    else:
        print("Take Physionet Basemodel trained on 6 channel pairs.")
        model_folder = "test_physio_34_10_65_90_101/2023-04-27_21-07-57_correctedBug/"

    source_model_path = os.path.join(root_path, model_directory+model_folder, "bestModel.h5")

    # FINETUNED MODEL
    model_save_path = os.path.join(root_path, "models/1D_CNN/"+CHANNEL_PAIRS+"/fine_tuned/physionet_base/")
    try:
        os.mkdir(model_save_path)
    except OSError as error:
        print(error)  
elif BASE_MODEL == "mydataset":
    # FINETUNED MODEL
    model_save_path = os.path.join(root_path, "models/1D_CNN/"+CHANNEL_PAIRS+"/fine_tuned/mydataset_base/correctedBug/")
    try:
        os.mkdir(model_save_path)
    except OSError as error:
        print(error)  
    

## Device settings
if GPU:
    # GPU setting
    tf.autograph.set_verbosity(0)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    # CPU setting
    tf.autograph.set_verbosity(0)
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print(physical_devices)


#Params
if CHANNEL_PAIRS == "ours_3_pairs":
    channels = [["C3","C4"],["F3","F4"],["P3","P4"]]
elif CHANNEL_PAIRS == "ours_6_pairs":
    channels = [["C3","C4"],["F3","F4"],["P3","P4"],["FP1","FP2"],["F7","F8"],["T3","T4"]]
else:
    print("Please select a valid channel pair setting!")
    channels = []

# Test subjects for testing transfer learning
test_subjects = ["dani", "ivo", "pablo", "huiyu", "manu", "fabio", "anna", "luisa", "sarah", "irene", "jan"]
# test_subjects = ["fabio", "irene", "jan"]
measurements = []


# Get dataset
trainingset = BCIDataset(data_root, test_subjects, [], measurement_length=4)
trainingset.validate_data()
trainingset.apply_bandpass_filtering(selected_data="sample")

log_accuracies = {}
for sub in test_subjects:
    log_accuracies[sub] = {'NotTuned': [], 'FineTuned': []}

    if BASE_MODEL == "mydataset":
        # SOURCE MODEL
        model_folder = "correctedBug/test_"+sub+"/correctedBug/"
        source_model_path = os.path.join(root_path, model_directory+model_folder, "bestModel.h5")

    #Load data
    x, y, _ = trainingset.load_subject_data(sub, channels)

    #Reshape for scaling
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    # #Grab a test set before SMOTE
    # x_train_raw, x_test_raw, y_train_raw, y_test = train_test_split(reshaped_x,
    #                                                                 y,
    #                                                                 stratify=y,
    #                                                                 test_size=0.20,
    #                                                                 random_state=42)

    # Get train-/test- indices of 5 fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(skf.split(reshaped_x, y)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        x_train_raw = reshaped_x[train_index]
        x_test_raw = reshaped_x[test_index]
        y_train_raw = y[train_index]
        y_test = y[test_index]


        #Scale indipendently train/test
        #Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.
        x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
        x_test_scaled_raw = minmax_scale(x_test_raw, axis=1)

        # x_test = x_test_scaled_raw.reshape(x_test_scaled_raw.shape[0], int(x_test_scaled_raw.shape[1]/2),2).astype(np.float64)

        x_test = x_test_scaled_raw.reshape(x_test_scaled_raw.shape[0], x.shape[1], x.shape[2]).astype(np.float64)
        x_test = np.swapaxes(x_test,1,2)

        #apply smote to train data
        from imblearn.over_sampling import SMOTE
        # Resample to specific number
        #sampling_class_counts = {'L': 400, 'R': 400}

        if CHANNEL_PAIRS == "ours_3_pairs":
            sampling_class_counts = {'L': 500, 'R': 500}
        elif CHANNEL_PAIRS == "ours_6_pairs":
            sampling_class_counts = {'L': 917, 'R': 917}
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

        # x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)
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
        y_test_01 = []
        for y_label in y_test:
            if y_label == 'L':
                y_test_01.append(0)
            elif y_label == 'R':
                y_test_01.append(1)
            else:
                print("Test Labels are different than L or R...")
        y_train = np.array(y_train_01)
        y_test = np.array(y_test_01)


        ## LOAD MODEL
        input_shape = (None, np.shape(x_train)[1], np.shape(x_train)[2])
            
        model = HopefullNet(inp_shape = (input_shape[1], input_shape[2]), two_class=True) # 500,2
        model.build(input_shape)
        model.load_weights(source_model_path)

        #Freeze conv layers
        for l in model.layers[:6]:
            l.trainable = False

        for l in model.layers:
            print(l._name, l.trainable)


        learning_rate = 1e-4 # default 1e-3
        loss = tf.keras.losses.binary_crossentropy
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Path of finetuned model    
        modelSubjFolder = os.path.join(model_save_path, 'sub_'+str(sub)+'/')
        try:
            os.mkdir(modelSubjFolder)
        except OSError as error:
            print(error)

        # Save detailed accuracy files and training history of fold
        foldCVPath = os.path.join(modelSubjFolder, 'fold_'+str(i)+'/')
        try:
            os.mkdir(foldCVPath)
        except OSError as error:
            print(error)

        modelname = "finetuned_sub_"+str(sub)+"_"+date+".h5"
        modelPath = os.path.join(foldCVPath, modelname)

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # Get accuracy before finetuning
        before_testLoss, before_testAcc = model.evaluate(x_test, y_test)


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
        

        hist = model.fit(x_train, y_train, epochs=20, batch_size=10,
                        validation_data=(x_test, y_test), callbacks=callbacksList) #32
        
        with open(os.path.join(foldCVPath, "hist_finetuned_sub_"+str(sub)+"_"+date+".pkl"), "wb") as file:
            pickle.dump(hist.history, file)


        testLoss, testAcc = model.evaluate(x_test, y_test)

        print('Before Accuracy:', before_testAcc)
        print('Before Loss: ', before_testLoss)

        print('After Accuracy:', testAcc)
        print('After Loss: ', testLoss)

        log_accuracies[sub]['NotTuned'].append(before_testAcc)      
        log_accuracies[sub]['FineTuned'].append(testAcc)

        from sklearn.metrics import classification_report, confusion_matrix
        # get list of MLP's prediction on test set
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
        f = open(os.path.join(foldCVPath,"classification_report_"+date+".pkl"),"wb")
        pickle.dump(classif_report,f)
        f.close()

        tn, fp, fn, tp = confusion_matrix(
                        yTestClass,
                        yPredClass,
                        ).ravel()
        conf_matrix = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
        print('\n Confusion matrix \n\n', conf_matrix)
        f = open(os.path.join(foldCVPath,"confusion_matrix_"+date+".pkl"),"wb")
        pickle.dump(conf_matrix,f)
        f.close()

# Save log_accuracies as file
f = open(os.path.join(model_save_path,"test_accuracies_"+date+".pkl"),"wb")
# write the python object (dict) to pickle file
pickle.dump(log_accuracies,f)
# close file
f.close()