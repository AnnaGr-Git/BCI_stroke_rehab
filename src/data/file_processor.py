import glob

import numpy as np
import scipy.io
import scipy.signal

from src.visualization.plotting import plot_logvar, plot_psd
from src.data.signal_processing import bandpass, logvar, apply_mix, psd, csp


def read_files(path):
    EEG_files = glob.glob(path)
    print(EEG_files)
    mat_raw = [scipy.io.loadmat(file) for file in EEG_files]
    print("Loading %s files..." % (len(EEG_files)))
    return mat_raw


class Data:

    def __init__(self, m):
        self.cl2 = None
        self.cl1 = None
        self.nsamples_win = None
        self.trials = None
        self.nevents = None
        self.nclasses = None
        self.cl_lab = None
        self.labels = None
        self.event_codes = None
        self.event_onsets = None
        self.nsamples = None
        self.nchannels = None
        self.EEG = None
        self.sample_rate = None
        self.channel_names = None
        self.m = m

        self.info_extract()
        self.trials_setup()

    def trials_setup(self):
        # Cut trials for the two classes in the interval 0.5-2.5s after the onset of the cue
        trials = {}
        self.cl1 = self.cl_lab[0]
        self.cl2 = self.cl_lab[1]
        # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
        win = np.arange(int(1 * self.sample_rate), int(5 * self.sample_rate))

        # Length of the time window
        self.nsamples_win = len(win)
        # Loop over the classes (right, foot)
        for cl, code in zip(self.cl_lab, np.unique(self.event_codes)):
            # Extract the onsets for the class
            cl_onsets = self.event_onsets[self.event_codes == code]
            # Allocate memory for the trials
            trials[cl] = np.zeros((self.nchannels, self.nsamples_win, len(cl_onsets)))
            # Extract each trial
            for i, onset in enumerate(cl_onsets):
                trials[cl][:, :, i] = self.EEG[:, win + onset]

        # Some information about the dimensionality of the data (channels x time x trials)
        print('Shape of trials[cl1]:', trials[self.cl1].shape)
        print('Shape of trials[cl2]:', trials[self.cl2].shape)

        self.trials = trials

    def info_extract(self):
        print("Extracting information...")
        self.sample_rate = self.m['nfo']['fs'][0][0][0][0]
        self.EEG = self.m['cnt'].T
        self.nchannels, self.nsamples = self.EEG.shape
        print(f"\t The EEG data presents {self.nchannels} channels and {self.nsamples} samples")
        self.channel_names = ["CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "FZ", "FP1", "FP2", "F7", "F3", "F4", "F8",
                              "T3", "C3"]
        self.event_onsets = self.m['mrk'][0][0][0]
        self.event_codes = self.m['mrk'][0][0][1]
        self.labels = np.zeros((1, self.nsamples), int)
        self.labels[0, self.event_onsets] = self.event_codes
        # 1 left hand -1 right hand
        #
        self.cl_lab = [s[0] for s in self.m['nfo']['classes'][0][0][0]]

        self.nclasses = len(self.cl_lab)
        self.nevents = len(self.event_onsets[0])
        print(f"\t The EEG data presents {self.nclasses} classes: {self.cl_lab} and {self.nevents} events")

    def print_info(self):
        # Print some information
        print("Printing information...")
        print('\t Shape of EEG:', self.EEG.shape)
        print('\t Sample rate:', self.sample_rate)
        print('\t Number of channels:', self.nchannels)
        print('\t Channel names:', self.channel_names)
        print('\t Number of events:', len(self.event_onsets[0]))
        print('\t Event codes:', np.unique(self.event_codes))
        print('\t Class labels:', self.cl_lab)
        print('\t Number of classes:', self.nclasses)

    def trial_filtering(self):

        trials_filt = {
            self.cl1: bandpass(self.trials[self.cl1], 8, 15, self.sample_rate, self.nchannels, self.nsamples_win),
            self.cl2: bandpass(self.trials[self.cl2], 8, 15, self.sample_rate, self.nchannels, self.nsamples_win)}
        return trials_filt

    def log_variation(self):
        trials_filt = self.trial_filtering()
        self.plot_log(trials_filt)

    def csp(self):
        trials_filt = self.trial_filtering()
        W = csp(trials_filt[self.cl2], trials_filt[self.cl1], self.nsamples_win)
        trials_csp = {self.cl1: apply_mix(W, trials_filt[self.cl1], self.nchannels, self.nsamples_win),
                      self.cl2: apply_mix(W, trials_filt[self.cl2], self.nchannels, self.nsamples_win)}
        self.plot_log(trials_csp)

        psd_l, freqs = psd(trials_csp[self.cl1], self.nchannels, self.nsamples_win, self.sample_rate)
        psd_r, freqs = psd(trials_csp[self.cl2], self.nchannels, self.nsamples_win, self.sample_rate)
        trials_PSD = {self.cl1: psd_l, self.cl2: psd_r}

        plot_psd(trials_PSD, freqs, [self.channel_names.index(ch) for ch in ['C3', 'CZ', 'C4']],
                 chan_lab=['first component', 'middle component', 'last component'],
                 maxy=0.8)

        # plot_scatter(trials_csp[cl1], trials_csp[cl2], self.cl_lab)

    def plot_log(self, trials):

        trials_logvar = {self.cl1: logvar(trials[self.cl1]),
                         self.cl2: logvar(trials[self.cl2])}
        plot_logvar(trials_logvar, self.nchannels, self.cl_lab)
        # plot_scatter(trials_logvar[cl1], trials_logvar[cl2], self.cl_lab)
