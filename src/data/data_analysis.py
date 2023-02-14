import pickle
import tkinter
from tkinter import simpledialog

import time
import pathlib
from pathlib import Path
import os

import numpy as np
from matplotlib import pyplot as plt

from src.data.data_processing import create_data, create_data_nn, validity
from src.data.file_processor import Data
from src.visualization.plotting import plot_psd, plot_LDA, plot_logvar, plot_before_after_csp
from src.data.signal_processing import bandpass, psd, logvar, apply_mix, csp, train_lda, accuracies
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

sample_rate = 125
channel_names = ["CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "FZ", "FP1", "FP2", "F7", "F3", "F4", "F8",
                 "T3", "C3"]


def plot_data(trials, title=None):
    psd_l, freqs = psd(trials["left"], n_channels, nsamples_win, sample_rate)
    psd_r, freqs = psd(trials["right"], n_channels, nsamples_win, sample_rate)

    trials_PSD = {"left": psd_l, "right": psd_r}

    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in
         ["C3", "T3", "CZ", "T4", "C4"]],
        # ["CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "FZ", "FP1", "FP2", "F7", "F3", "F4", "F8",
        #  "T3", "C3"]],
        chan_lab=["C3", "T3", "CZ", "T4", "C4"],
        title=title
    )

    # to_try = np.swapaxes(trials_PSD["left"][15,:,:],1,0)
    # maximum = []
    # for trials in to_try:
    #     maximum.append(max(trials))
    #     plt.plot(freqs, trials, label="left")
    #     #plt.show()


def filtering_data(trials_f):
    trials_filtered = {"left": bandpass(trials_f["left"], 8, 15, sample_rate, n_channels, nsamples_win),
                       "right": bandpass(trials_f["right"], 8, 15, sample_rate, n_channels, nsamples_win)}

    return trials_filtered


def log_variation(trials_l):
    trials_logvar = {"left": logvar(trials_l["left"]),
                     "right": logvar(trials_l["right"])}

    # y0 = np.mean(trials_logvar["left"], axis=1)
    # y1 = np.mean(trials_logvar["right"], axis=1)
    # diff = []
    # for i, y in enumerate(y0):
    #     diff.append(y - y1[i])
    #
    # x0 = np.arange(len(diff))
    #
    # plt.figure(figsize=(12, 5))
    # lft = np.zeros(len(diff))
    # rght = np.zeros(len(diff))
    #
    # for i, value in enumerate(diff):
    #     if value < 0:
    #         rght[i] = value
    #     else:
    #         lft[i] = value
    #
    #
    # plt.bar(x0, rght, width=0.5, color="r")
    # plt.bar(x0, lft, width=0.5, color="b")
    # plt.xlim(-0.5, len(diff) + 0.5)
    # plt.title("Difference of LogVar before CSP")
    # plt.xlabel("Components")
    # plt.ylabel("LogVariance")
    # plt.legend(["right", "left"])
    # plt.gca().yaxis.grid(True)
    # plt.show()

    # plot_logvar(trials_logvar, n_channels, ["left", "right"])
    return trials_logvar


def csp_data(trials_csp, title=None):
    W_mtx = csp(trials_csp["left"], trials_csp["right"], nsamples_win)
    trials = {"left": apply_mix(W_mtx, trials_csp["left"], n_channels, nsamples_win),
              "right": apply_mix(W_mtx, trials_csp["right"], n_channels, nsamples_win)}
    # psd_l, freqs = psd(trials["left"], n_channels, nsamples_win, sample_rate)
    # psd_r, freqs = psd(trials["right"], n_channels, nsamples_win, sample_rate)

    # trials_PSD = {"left": psd_l, "right": psd_r}

    # plot_psd(
    #     trials_PSD,
    #     freqs,
    #     np.arange(16),
    #     title=title
    # )
    #
    # plot_psd(
    #     trials_PSD,
    #     freqs,
    #     [0, -1],
    #     chan_lab=["First", "Last"],
    #     title="Train Data After CSP First and Last PC"
    # )
    # trials_log_var = log_variation(trials_PSD)
    # c1, c2 = best_components(trials_log_var)
    # # Select only the first and last components for classification
    # comp = [c1, c2]
    # plot_psd(
    #     trials_PSD,
    #     freqs,
    #     comp,
    #     chan_lab=["PC " + str(c1), "PC " + str(c2)],
    #     title="Train Data After CSP Best PC"
    # )

    return trials, W_mtx


def best_components(trials_log):
    y0 = np.mean(trials_log["left"], axis=1)
    y1 = np.mean(trials_log["right"], axis=1)
    diff = []
    for i, y in enumerate(y0):
        diff.append(y - y1[i])

    # plt.figure(figsize=(12, 5))
    # lft = np.zeros(len(diff))
    # rght = np.zeros(len(diff))
    #
    # for i, value in enumerate(diff):
    #     if value > 0:
    #         rght[i] = value
    #     else:
    #         lft[i] = value
    #
    # x0 = np.arange(len(diff))

    # plt.bar(x0, rght, width=0.5, color="r")
    # plt.bar(x0, lft, width=0.5, color="b")
    # plt.xlim(-0.5, len(diff) + 0.5)
    # plt.title("Difference of LogVar after CSP")
    # plt.xlabel("Components")
    # plt.ylabel("LogVariance")
    # plt.legend(["right", "left"])
    # plt.gca().yaxis.grid(True)
    # plt.show()

    pc_right = np.where(diff == np.amax(diff))[0]
    pc_left = np.where(diff == np.amin(diff))[0]
    return int(pc_right), int(pc_left)


def LDA_data(trials_data, train_percentage=0.95):
    # Percentage of trials to use for training (50-50 split here)

    # Calculate the number of trials for each class the above percentage boils down to
    n_train_l = int(trials_data["left"].shape[2] * train_percentage)
    n_train_r = int(trials_data["right"].shape[2] * train_percentage)
    train_data = {"left": trials_data["left"][:, :, :n_train_l],
                  "right": trials_data["right"][:, :, :n_train_r]}

    test_data = {"left": trials_data["left"][:, :, n_train_l:],
                 "right": trials_data["right"][:, :, n_train_r:]}

    # train_left, test_left = train_test_split(np.swapaxes(trials_data["left"], 2, 0), train_size=train_percentage)
    # train_right, test_right = train_test_split(np.swapaxes(trials_data["right"], 2, 0), train_size=train_percentage)
    #
    # # Splitting the frequency filtered signal into a train and test set
    # train_data = {"left": np.swapaxes(train_left, 0, 2),
    #               "right": np.swapaxes(train_right, 0, 2)}
    #
    # test_data = {"left": np.swapaxes(test_left, 0, 2),
    #              "right": np.swapaxes(test_right, 0, 2)}
    train_data_mix, train_matrix = csp_data(train_data, title="Train Data After CSP")

    test_data_mix = {"left": apply_mix(train_matrix, test_data["left"], n_channels, nsamples_win),
                     "right": apply_mix(train_matrix, test_data["right"], n_channels, nsamples_win)}

    trials_log_var = log_variation(train_data_mix)
    c1, c2 = best_components(trials_log_var)
    # Select only the first and last components for classification
    comp = np.array([c1, c2])
    train_best = {"left": train_data_mix["left"][comp, :, :],
                  "right": train_data_mix["right"][comp, :, :]}
    test_best = {"left": test_data_mix["left"][comp, :, :],
                 "right": test_data_mix["right"][comp, :, :]}

    # Calculate the log-var
    train_log = {"left": logvar(train_best["left"]),
                 "right": logvar(train_best["right"])}
    test_log = {"left": logvar(test_best["left"]),
                "right": logvar(test_best["right"])}

    return train_log, test_log, train_matrix, [c1, c2]


def norm_log_var(trial):
    t_var = []
    for t in trial:
        t_var.append(np.var(t, axis=0))

    output = []
    sum_var = sum(t_var)
    for i in range(trial.shape[0]):
        output.append(np.log(t_var[i] / sum_var))

    return np.array(output)

parent_path = str(Path().parent.resolve().as_posix())+"/"
data_path = os.path.join(parent_path, "data/raw/anna/2023-02-13/")
data = create_data(data_path)
trials_valid = validity(data)

n_channels = np.asarray(trials_valid["left"]).shape[1]
nsamples_win = np.asarray(trials_valid["left"]).shape[2]
nsize_left = np.asarray(trials_valid["left"]).shape[0]
nsize_right = np.asarray(trials_valid["right"]).shape[0]
#
left_trials = np.asarray(trials_valid["left"])
left_trials = np.swapaxes(np.swapaxes(left_trials, 1, 0), 2, 1)
right_trials = np.asarray(trials_valid["right"])
right_trials = np.swapaxes(np.swapaxes(right_trials, 1, 0), 2, 1)
#
trials_raw = {"left": left_trials, "right": right_trials}
# #
# m = scipy.io.loadmat('/home/nurife/BCI/IM/DataSets/SPC_Data_csv/GoodData/record05.mat',
#                     struct_as_record=True)  # Load matlab file
#
# data = Data(
# m)
# n_channels = data.nchannels
# nsamples_win = data.nsamples_win
# trials_raw = data.trials


plot_data(trials_raw, title="Raw Trials")
trials_filt = filtering_data(trials_raw)
plot_data(trials_filt, title="Filtered Trials")
log_variation(trials_filt)
trials_csp, _ = csp_data(trials_filt)
plot_data(trials_csp, title="Trials after CSP")
log_variation(trials_csp)

# LDA
train, test, W_matrix, pc = LDA_data(trials_filt, train_percentage=0.95)

W, b = train_lda(train["left"].T, train["right"].T)

print("Train accuracies")
accuracies(train, W, b)
print("Test accuracies")
accuracies(test, W, b)
plot_LDA(train, test, b, W, ["left", "right"], pc)

# before_csp = {"left": trials_raw["left"][pc, :, :],
#               "right": trials_raw["right"][pc, :, :]}
# plot_before_after_csp(before_csp, train, ["left", "right"], pc)

combined_data_train = create_data_nn(train)
combined_data_test = create_data_nn(test)

#
train_X = []
train_y = []
for X, y in combined_data_train:
    train_X.append(X)
    train_y.append(y)

test_X = []
test_y = []
for X, y in combined_data_test:
    test_X.append(X)
    test_y.append(y)
clf = LDA()
X_r2 = clf.fit(np.array(train_X), np.array(train_y)[:, 0])
score = clf.score(np.array(train_X), np.array(train_y)[:, 0])
# pred = clf.predict(test_X)
score = clf.score(test_X, np.array(test_y))

qlf = QDA()
qlf.fit(train_X, train_y)
pred_2 = qlf.predict_proba(test_X)
score_2 = qlf.score(test_X, np.array(test_y))

# Neural Network

# combined_data_train = create_data_nn(train)
# combined_data_test = create_data_nn(test)
# #
# train_X = []
# train_y = []
# for X, y in combined_data_train:
#     train_X.append(X)
#     train_y.append(y)
#
# test_X = []
# test_y = []
# for X, y in combined_data_test:
#     test_X.append(X)
#     test_y.append(y)
#
# train_X = np.array(train_X)
# train_y = np.array(train_y)
# test_X = np.array(test_X)
# test_y = np.array(test_y)
#
# # sc = StandardScaler()
# #
# # X_train = sc.fit_transform(train_X)
# # X_test = sc.transform(test_X)
#
# val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, shuffle=True)
#
# model = NN_model(train_X)
# epochs = 80
# batch_size = 3
# history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs,
#                     shuffle=True,
#                     validation_data=(val_X, val_y), validation_batch_size=1)
# fig = plot_history(history)
#
# score = model.evaluate(test_X, test_y, batch_size=batch_size)
# print(test_X.shape, test_y.shape)
# _pred = model.predict(test_X)
# accuracy = accuracy_score(test_y, np.rint(_pred))

ROOT = tkinter.Tk()

ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Save",
                                  prompt="Do you want to save? (y/n):")

if USER_INP == "y":
    time = int(time.time())

    # print("NN Accuracy: %.2f%%" % (accuracy * 100.0))
    # MODEL_NAME = f"models_09/NN/{round(score[1] * 100, 2)}-{epochs}epoch-{time}-loss-{round(score[0], 2)}.model"
    # model.save(MODEL_NAME)

    matrix = {'W_matrix': W_matrix}
    with open(f"models_09/W_matrix/{time}.pickle", "wb") as file:
        pickle.dump(matrix, file, protocol=pickle.HIGHEST_PROTOCOL)

    data = {'W': W, 'b': b}
    # with open(f"models_09/LDA/{round(left / total_l * 100, 1)}_r_{round(right / total_r * 100, 2)}_acc_{time}.pickle",
    #           "wb") as file:
    #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    # fig.savefig(f'models_09/NN/{time}-loss_plot.png')

    print("Model Saved")
