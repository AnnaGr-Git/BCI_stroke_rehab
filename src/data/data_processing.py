import os

import numpy as np
import pandas as pd

ACTIONS = ["left", "right"]


def create_data(starting_dir):
    print(starting_dir)
    training_files = {}
    directory = os.listdir(starting_dir)

    for action in ACTIONS:
        training_files[action] = [starting_dir + file for file in directory if (action in file)]

    lengths = [len(training_files[action]) for action in ACTIONS]
    print(f"Number of files {lengths}")

    training_data = {}
    for action in ACTIONS:
        files = [pd.read_csv(os.path.abspath(file)) for file in training_files[action]]
        training_data[action] = [data.to_numpy()[:, 3:].transpose() for data in files]

    data_lengths = []
    # for action in ACTIONS:
    #    data_lengths += ([len(data[1]) for data in training_data[action]])
    data_length = 420
    final_data = {"left":[], "right":[]}
    for action in ACTIONS:
        for data in training_data[action]:
            if data.shape[1] > (data_length - 1):
                final_data[action].append(data[:, 125:data_length])

    lengths = [final_data[action][0].shape for action in ACTIONS]
    print(f"Shape of the training data: {lengths}")

    return final_data


def create_data_nn(data):
    # creating X, y
    combined_data = []
    data["left"] = data["left"].transpose()
    data["right"] = data["right"].transpose()
    for action in ACTIONS:
        for trial in data[action]:

            if action == "left":
                combined_data.append([trial, np.array([1])])

            elif action == "right":
                # np.append(combined_data, np.array([data, [1, 0]]))
                combined_data.append([trial, np.array([0])])

    np.random.shuffle(combined_data)
    print("Total data length:", len(combined_data))
    return combined_data

def validity(trials):
    valid_trials = {"left": [], "right": []}
    invalid = 0
    volt = 250
    for epoch in trials["left"]:
        max_R = np.array([max(epoch[i, :]) for i in range(16)])
        min_R = np.array([min(epoch[i, :]) for i in range(16)])
        if any(max_R > volt) or any(min_R < -volt):
            invalid += 1
        else:
            valid_trials["left"].append(epoch)
    for epoch in trials["right"]:
        max_R = np.array([max(epoch[i, :]) for i in range(16)])
        min_R = np.array([min(epoch[i, :]) for i in range(16)])
        if any(max_R > volt) or any(min_R < -volt):
            invalid += 1
        else:
            valid_trials["right"].append(epoch)

    print(f"Total invalid trials {invalid}")
    return valid_trials