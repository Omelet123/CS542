import numpy as np
from numpy import linalg
import scipy.io
from pandas_ml import ConfusionMatrix
from SVM import SVM
from join import join_cluster

mat = scipy.io.loadmat('MNIST_data.mat')
x_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))
x_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))


def data_clustering(x_train, y_train):
    x_t = [[] for i in range(10)]

    for i in range(x_train.shape[0]):
        x_t[y_train[i]].append(x_train[i])

    return np.array(x_t[0]), np.array(x_t[1]), np.array(x_t[2]), np.array(x_t[3]), np.array(x_t[4]), np.array(x_t[5]), np.array(x_t[6]), np.array(x_t[7]), np.array(x_t[8]), np.array(x_t[9])


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def one_vs_all():
    x_t0, x_t1, x_t2, x_t3, x_t4, x_t5, x_t6, x_t7, x_t8, x_t9 = data_clustering(x_train, y_train)
    numpy_predict = []

    for number in range(10):
        train_number, train_rest, test_number, test_rest = join_cluster(x_t0, x_t1, x_t2, x_t3, x_t4, x_t5, x_t6, x_t7, x_t8, x_t9, number)
        training_data = np.vstack((train_number, train_rest))
        test_data = np.hstack((test_number, test_rest))
        clf = SVM(kernel=polynomial_kernel, C=0.1)
        clf.train(training_data, test_data)
        y_predict = clf.compute(x_test)
        numpy_predict.append(y_predict)

    prediction = np.argmax(np.array(numpy_predict), axis = 0 )
    correct = np.sum(prediction == y_test)
    confusion_matrix = ConfusionMatrix(y_test, prediction)
    size = len(y_predict)
    accuracy = (correct/float(size)) * 100
    print("Confusion matrix:\n%s" % confusion_matrix)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print("The accuracy is  ", accuracy, "%")


one_vs_all()
