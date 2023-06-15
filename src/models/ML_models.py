import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA



def NN_model(X_train):
    NN_mod = Sequential()
    # The Input Layer :
    NN_mod.add(Dense(16, kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'))

    # The Hidden Layer :
    NN_mod.add(Dense(8, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_mod.add(Dense(1))
    NN_mod.add(Activation('sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0006)

    # Compile the network :
    NN_mod.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=['accuracy'])
    return NN_mod

def LDA_model():
    return LDA()

def QDA_model():
    return QDA()

def XGB_model():
    model = XGBClassifier(objective='binary:logistic', max_depth=5)
    return model

def SVM_model():
    return SVC(gamma='auto')

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