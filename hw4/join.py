import numpy as np
from numpy import linalg
import scipy.io
from pandas_ml import ConfusionMatrix
from SVM import SVM

def join_cluster(X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9, number):

    if number == 0:
        X_train_rest = np.vstack((X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train0 = np.ones(len(X_train0))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train0 , X_train_rest, y_train0, y_train_rest

    elif number == 1:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train1 = np.ones(len(X_train1))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train1 , X_train_rest, y_train1, y_train_rest

    elif number == 2:
        X_train_rest = np.vstack((X_train0, X_train1, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train2 = np.ones(len(X_train2))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train2 , X_train_rest, y_train2, y_train_rest

    elif number == 3:
        X_train_rest = np.vstack((X_train0, X_train2, X_train1, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train3 = np.ones(len(X_train3))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train3 , X_train_rest, y_train3, y_train_rest

    elif number == 4:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train1, X_train5, X_train6, X_train7, X_train8, X_train9))
        y_train4 = np.ones(len(X_train4))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train4 , X_train_rest, y_train4, y_train_rest

    elif number == 5:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train1, X_train6, X_train7, X_train8, X_train9))
        y_train5 = np.ones(len(X_train5))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train5 , X_train_rest, y_train5, y_train_rest

    elif number == 6:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train1, X_train7, X_train8, X_train9))
        y_train6 = np.ones(len(X_train6))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train6 , X_train_rest, y_train6, y_train_rest

    elif number == 7:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train1, X_train8, X_train9))
        y_train7 = np.ones(len(X_train7))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train7 , X_train_rest, y_train7, y_train_rest

    elif number == 8:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train1, X_train9))
        y_train8 = np.ones(len(X_train8))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train8 , X_train_rest, y_train8, y_train_rest

    elif number == 9:
        X_train_rest = np.vstack((X_train0, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train1))
        y_train9 = np.ones(len(X_train9))
        y_train_rest = np.ones(len(X_train_rest)) * -1
        return X_train9 , X_train_rest, y_train9, y_train_rest
