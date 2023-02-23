import numpy as np
import scipy
from matplotlib import mlab
from scipy import linalg, integrate
from scipy.signal import freqs

def psd(dataset, selected_data:str='sample'):
    """
    Calculates for each trial the Power Spectral Density (PSD).

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.
    freqs : list of floats
        Yhe frequencies for which the PSD was computed (useful for plotting later)
        :param nsamples:
        :param sample_rate:
        :param trials:
        :param nchannels:
    """

    ntrials = len(dataset)
    nchannels = len(dataset.channel_names)
    ndatapoints = dataset.data_length
    trials_PSD = np.zeros((ntrials, 251, nchannels))

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            sample_values_ch = dataset.get_sample_values(trial, selected_data=selected_data)[:,ch]
            (PSD, frequency) = mlab.psd(sample_values_ch, NFFT=int(ndatapoints), Fs=dataset.sample_rate)
            trials_PSD[trial, :, ch] = PSD.ravel()

    return trials_PSD, frequency

def cov(trials, nsamples):
    """ Calculate the covariance for each trial and return their average """
    ntrials = trials.shape[2]
    covariances = [trials[:, :, i].dot(trials[:, :, i].T) / nsamples for i in range(ntrials)]
    return np.mean(covariances, axis=0)


def whitening(sigma):
    """ Calculate a whitening matrix for covariance matrix sigma. """
    U, l, _ = linalg.svd(sigma)
    return U.dot(np.diag(l ** -0.5))


def bandpass(trials, lo, hi, sample_rate, nchannels, nsamples):
    """
    Designs and applies a bandpass filter to the signal.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)

    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
    """

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the nyquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial

    ntrials = trials.shape[2]

    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt


def logvar(trials):
    """
    Calculate the log-var of each channel.

    Parameters
    ----------
    trials : 3d-array (trials x samples x channels)
        The EEG signal.

    Returns
    -------
    jmnar - 2d-array (trials x channels)
        For each channel the logvar of the signal
    """
    return np.log(np.var(trials, axis=1))


def csp(trials_r, trials_l, nsamples):
    """
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right-hand movement trials
        trials_l - Array (channels x samples x trials) containing left-hand movement trials
    returns:
        Mixing matrix W
    """

    cov_r = cov(trials_r, nsamples)
    cov_l = cov(trials_l, nsamples)

    P = whitening(cov_r + cov_l)

    B, _, _ = linalg.svd(P.T.dot(cov_l).dot(P))
    W = P.dot(B)
    # spatial filters (maximise one variance, minimise other)
    return W


def apply_mix(W, trials, nchannels, nsamples):
    """ Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)"""
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:, :, i] = W.T.dot(trials[:, :, i])
    return trials_csp

def best_csp_components(trials_csp, classes:list=['arm_left', 'arm_right']):
    # Calculate log-var of arrays
    logvar_class1 = logvar(trials_csp[classes[0]])
    logvar_class2 = logvar(trials_csp[classes[1]])

    # Get mean values for each component over all trials    
    y0 = np.mean(logvar_class1, axis=0)
    y1 = np.mean(logvar_class2, axis=0)
    # Get difference of logvar values per component (class1-class2)
    diff = []
    for i, y in enumerate(y0):
        diff.append(y - y1[i])

    # Choose component where difference is max and min
    best_max = int(np.where(diff == np.amax(diff))[0])
    best_min = int(np.where(diff == np.amin(diff))[0])
    return [best_max, best_min]


def integration(y_data=None, chan_lab=None, cl=None, trials_PSD=None, ch=None):
    cl = list(trials_PSD.keys())[0]
    y_data = np.mean(trials_PSD[cl][ch, :, :], axis=1)

    integration = integrate.trapz(y_data, freqs)
    print('Channel %s, Class %s >> %f' % (chan_lab, cl, integration))
    return integration


def train_lda(class1, class2):
    '''
    Trains the LDA algorithm.
    arguments:
        class1 - An array (observations x features) for class 1
        class2 - An array (observations x features) for class 2
    returns:
        The projection matrix W
        The offset b
    '''
    nclasses = 2

    nclass1 = class1.shape[0]
    nclass2 = class2.shape[0]

    # Class priors: in this case, we have an equal number of training
    # examples for each class, so both priors are 0.5
    prior1 = nclass1 / float(nclass1 + nclass2)
    prior2 = nclass2 / float(nclass1 + nclass1)

    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)

    class1_centered = class1 - mean1
    class2_centered = class2 - mean2

    # Calculate the covariance between the features
    cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
    cov2 = class2_centered.T.dot(class2_centered) / (nclass2 - nclasses)

    W = (mean2 - mean1).dot(np.linalg.pinv(prior1 * cov1 + prior2 * cov2))
    b = (prior1 * mean1 + prior2 * mean2).dot(W)

    return W, b


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


def accuracies(trials_acc, W_value, b_value):
    left_wrong = 0
    left_correct = 0
    right_wrong = 0
    right_correct = 0

    x = np.array([-5, 1])
    y = (b_value - W_value[0] * x) / W_value[1]

    a = np.array([x[0], y[0]])
    c = np.array([x[1], y[1]])

    lt = trials_acc["left"].transpose()
    rt = trials_acc["right"].transpose()

    for point in lt:
        results = np.cross(point - a, c - a)
        if results > 0:
            left_correct += 1
        else:
            left_wrong += 1

    for point in rt:
        results = np.cross(point - a, c - a)
        if results < 0:
            right_correct += 1
        else:
            right_wrong += 1

    #print(f"Left {left_correct / len(lt)} Right: {right_correct / len(rt)}")

    total = float((left_correct / (len(lt)) + right_correct / (len(rt))) / 2)
    return total
