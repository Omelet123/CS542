from numpy import linalg
import scipy.io
import numpy as np
import itertools
from pandas_ml import ConfusionMatrix
from SVM import SVM

# load data from training and  testing data sets with label
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


def generate_data(x1, x2):
    y1 = np.ones(len(x1))
    y2 = np.ones(len(x2)) * -1

    return y1, y2


def transform_data(predict, plus, minus):
    for i in range(predict.shape[0]):
        if predict[i] == 1:
            predict[i] = plus
        elif predict[i] == -1:
            predict[i] = minus

    return predict


def decision_tree(classes, data_values):
    number = classes
    combination = list(itertools.combinations(classes, 2))

    if len(combination) > 1:
        if data_values[combination[0]] < 0:
            number.pop(0)
            return decision_tree(number, data_values)
        else:
            number.pop(1)
            return decision_tree(number, data_values)
    elif len(combination) == 1:
        if data_values[combination[0]] < 0:
            return combination[0][1]
        else:
            return combination[0][0]


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def one_vs_one():
    x_t0, x_t1, x_t2, x_t3, x_t4, x_t5, x_t6, x_t7, x_t8, x_t9 = data_clustering(x_train, y_train)
    prediction = []
    numpy_list = []
    numpy_predict = [x_t0, x_t1, x_t2, x_t3, x_t4, x_t5, x_t6, x_t7, x_t8, x_t9]
    combination = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

    for pair in combination:

        y1, y2 = generate_data(numpy_predict[pair[0]], numpy_predict[pair[1]])

        training_data = np.vstack((numpy_predict[pair[0]] , numpy_predict[pair[1]]))
        test_data = np.hstack((y1, y2))

        clf = SVM(kernel=polynomial_kernel, C=0.1)
        clf.train(training_data, test_data)

        y_predict = clf.compute(x_test)
        numpy_list.append(y_predict)

    numpy_list = np.array(numpy_list)
    transpose = np.transpose(numpy_list)
    mix = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

    for row in transpose:
        newdict = {}
        for i in range(len(mix)):
            newdict[mix[i]] = row[i]

        result = decision_tree([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], newdict)
        prediction.append(result)
    prediction = np.array(prediction)
    correct = np.sum(prediction == y_test)
    confusion_matrix = ConfusionMatrix(y_test, prediction)
    size = len(y_predict)
    accuracy = (correct/float(size)) * 100
    print("Confusion matrix:\n%s" % confusion_matrix)
    print("The accuracy is  ", accuracy, "%")


one_vs_one()
