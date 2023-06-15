import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
from src.data.signal_processing import logvar, psd


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
            plt.plot(freqs, np.mean(trials_PSD[cl][:, :, ch], axis=0), label=cl)

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

def plot_data(dataset, title: str="", channels: list=["C3", "T3", "CZ", "T4", "C4"], selected_data:str='sample'):
    # Calc psd
    psd_set, freqs = psd(dataset, selected_data)
    print(np.shape(psd_set))

    # Get psd for each class
    trials_PSD = {}
    for cl in dataset.classes:
        class_indexes = list(np.where(dataset.data["class"] == cl)[0])
        trials_PSD[cl] = psd_set[class_indexes,:,:]

    #trials_PSD = {'arm_left': psd_set[class_indexes['arm_left'],:,:], 'arm_right': psd_set[class_indexes['arm_right'],:,:]}

    # Plot mean PSD of selected channels
    plot_psd(
        trials_PSD,
        freqs,
        [dataset.channel_names.index(ch) for ch in channels],
        chan_lab=channels,
        title=title
    )
    


def plot_logvar(trials, classes, title):
    """
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    """
    plt.figure(figsize=(12, 5))

    y0 = np.mean(trials[classes[0]], axis=0)
    y1 = np.mean(trials[classes[1]], axis=0)
    x0 = np.arange(len(y0)) - 0.2
    x1 = np.arange(len(y1)) + 0.2

    plt.bar(x0, y0, width=0.4, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, len(y0) + 0.5)

    plt.gca().yaxis.grid(True)
    plt.title('Log-Var of each component '+title)
    plt.xlabel('Components')
    plt.ylabel("Log Variance")
    plt.legend(classes)
    plt.show()

def plot_logvar_diff(trials, classes, title):
    """
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    """
    y0 = np.mean(trials[classes[0]], axis=0)
    y1 = np.mean(trials[classes[1]], axis=0)
    diff = []
    for i, y in enumerate(y0):
        diff.append(y - y1[i])
    
    x0 = np.arange(len(diff))
    
    plt.figure(figsize=(12, 5))
    lft = np.zeros(len(diff))
    rght = np.zeros(len(diff))
    
    for i, value in enumerate(diff):
        if value < 0:
            rght[i] = value
        else:
            lft[i] = value
    
    
    plt.bar(x0, rght, width=0.5, color="r")
    plt.bar(x0, lft, width=0.5, color="b")
    plt.xlim(-0.5, len(diff) + 0.5)
    plt.title("Difference of Log-Var "+title)
    plt.xlabel("Components")
    plt.ylabel("Log Variance")
    plt.legend(classes)
    plt.gca().yaxis.grid(True)
    plt.show()

def plot_data_logvar(dataset, selected_data, mode:str="diff", classes:list = ['arm_left', 'arm_right'], title:str=""):
    # Get data as array
    data_arr = dataset.get_data_array(selected_data=selected_data)
    data_logvar = logvar(data_arr)

    # Get data of each class
    class_indexes = {classes[0]:[], classes[1]:[]}
    for cl in classes:
        class_indexes[cl] = list(np.where(dataset.data["class"] == cl)[0])

    trials_logvar = {classes[0]: data_logvar[class_indexes[classes[0]],:], classes[1]: data_logvar[class_indexes[classes[1]],:]}

    if mode == "single":
        plot_logvar(trials_logvar, classes, title)
    elif mode == "diff":
        plot_logvar_diff(trials_logvar, classes, title)
    else:
        plot_logvar(trials_logvar, classes, title)
        plot_logvar_diff(trials_logvar, classes, title)



def plot_scatter(data_cl1, data_cl2, classes, pc=None):
    if pc is None:
        pc = [0, 1]
    plt.scatter(data_cl1[:, 0], data_cl1[:, -1], color='b')
    plt.scatter(data_cl2[:, 0], data_cl2[:, -1], color='r')
    plt.xlabel(f'Component:{pc[0]}')
    plt.ylabel(f'Component:{pc[1]}')
    plt.legend(classes)


def plot_LDA(train, test, b, W, classes, components=None):
    # Scatterplot like before
    pc = [components[0], components[-1]]
    cl1 = classes[0]
    cl2 = classes[1]
    # Calculate decision boundary (x,y)
    min_val1 = np.min(train[cl1])
    min_val2 = np.min(train[cl2])
    min_val = np.min([min_val1, min_val2])
    max_val1 = np.max(train[cl1])
    max_val2 = np.max(train[cl2])
    max_val = np.max([max_val1, max_val2])
    print(f"MIN: {min_val}")
    
    # x = np.arange(-3,1.5, 0.1)
    x = np.arange(min_val,max_val,0.1)
    y = (b - W[0] * x) / W[-1]

    print(f"Components: {pc}")
    print(f"Class 1: {cl1}")
    print(f"Class 2: {cl2}")

    print(f"shape: {np.shape(train[cl1])}")

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plot_scatter(train[cl1], train[cl2], classes, pc)
    plt.title('Training data')
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    #plt.ylim(-2.5, 1)

    plt.subplot(1, 2, 2)
    plot_scatter(test[cl1], test[cl2], classes, pc)
    plt.title('Test data')
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    #plt.ylim(-2.5, 1)

    plt.show()

# def plot_before_after_csp(before_csp, after_csp, cl_lab, pc = None):
#     # Scatterplot like before
#     if pc is None:
#         pc = [0, 1]
#     cl1 = cl_lab[0]
#     cl2 = cl_lab[1]


#     plt.subplot(1, 2, 1)
#     plot_scatter(before_csp[cl1], before_csp[cl2], cl_lab,pc)
#     plt.title('Before CSP')

#     plt.subplot(1, 2, 2)
#     plot_scatter(after_csp[cl1], after_csp[cl2], cl_lab,pc)
#     plt.title('After CSP')

#     plt.show()


# def plot_history(history):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.plot(history.history['accuracy'], label='accuracy')
#     ax1.plot(history.history['val_accuracy'], label='val_accuracy')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Accuracy')
#     #ax1.set_ylim([0, 2])
#     ax1.legend(loc='lower right')

#     ax2.plot(history.history['loss'], label='loss')
#     ax2.plot(history.history['val_loss'], label='val_loss')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Loss')
#     #ax2.set_ylim([0., 2])
#     ax2.legend(loc='upper right')

#     plt.show()
#     return fig