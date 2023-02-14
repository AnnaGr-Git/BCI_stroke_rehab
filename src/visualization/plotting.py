import numpy as np
from matplotlib import pyplot as plt


def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None, title=None):
    """
    Plots PSD data calculated with psd().

    Parameters
    ----------
    trials_PSD : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd()
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    """

    plt.figure(figsize=(12, 5))

    nchans = len(chan_ind)

    # Maximum of 3 plots per row
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)

    # Enumerate over the channels
    for i, ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows, ncols, i + 1)
        #plt.tight_layout()
        # Plot the PSD for each class
        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch, :, :], axis=1), label=cl)

        # All plot decoration below...

        plt.xlim(1, 30)

        if maxy is not None:
            plt.ylim(0, maxy)

        plt.grid()

        plt.xlabel('Frequency (Hz)')

        if chan_lab is None:
            plt.title('Channel %d' % (ch + 1))
        else:
            plt.title(chan_lab[i])
        #plt.supylabel("Power Spectrum (W)")
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_logvar(trials, nchannels, cl_lab):
    """
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    """
    plt.figure(figsize=(12, 5))

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4

    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels + 0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(cl_lab)
    plt.show()


def plot_scatter(left, right, cl_lab, pc=None):
    # plt.figure()
    if pc is None:
        pc = [0, 1]
    plt.scatter(left[0, :], left[-1, :], color='b')
    plt.scatter(right[0, :], right[-1, :], color='r')
    plt.xlabel(f'Component:{pc[0]}')
    plt.ylabel(f'Component:{pc[1]}')
    plt.legend(cl_lab)


def plot_LDA(train, test, b, W, cl_lab, pc=None):
    # Scatterplot like before
    if pc is None:
        pc = [0, 1]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    # Calculate decision boundary (x,y)
    x = np.arange(-3,1.5, 0.1)
    y = (b - W[0] * x) / W[1]

    plt.subplot(1, 2, 1)
    plot_scatter(train[cl1], train[cl2], cl_lab,pc)
    plt.title('Training data')
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    plt.ylim(-2.5, 1)

    plt.subplot(1, 2, 2)
    plot_scatter(test[cl1], test[cl2], cl_lab,pc)
    plt.title('Test data')
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    plt.ylim(-2.5, 1)

    plt.show()

def plot_before_after_csp(before_csp, after_csp, cl_lab, pc = None):
    # Scatterplot like before
    if pc is None:
        pc = [0, 1]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]


    plt.subplot(1, 2, 1)
    plot_scatter(before_csp[cl1], before_csp[cl2], cl_lab,pc)
    plt.title('Before CSP')

    plt.subplot(1, 2, 2)
    plot_scatter(after_csp[cl1], after_csp[cl2], cl_lab,pc)
    plt.title('After CSP')

    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    #ax1.set_ylim([0, 2])
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    #ax2.set_ylim([0., 2])
    ax2.legend(loc='upper right')

    plt.show()
    return fig