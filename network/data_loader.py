import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import scipy.io as sio
from network.constants import DataType


def load_data_mlp(dataType):
    if dataType == DataType.MNIST:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
    elif dataType == DataType.SVHN:
        train_data = sio.loadmat('data/train_32x32.mat')
        test_data = sio.loadmat('data/test_32x32.mat')

        X_train = train_data['X']
        y_train = train_data['y'].reshape(-1)
        y_train = np.array(list(map(fix_0, y_train)))
        X_test = test_data['X']
        y_test = test_data['y'].reshape(-1)
        y_test = np.array(list(map(fix_0, y_test)))
        X_train = X_train.reshape(3072, 73257)
        X_test = X_test.reshape(3072, 26032)
        X_train = np.swapaxes(X_train, 0, 1)
        X_test = np.swapaxes(X_test, 0, 1)
    elif dataType == DataType.CIFAR:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


def fix_0(digit):
    if digit == 10:
        return 0
    else:
        return digit


def load_data_cnn(dataType):
    if dataType == DataType.MNIST:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
    elif dataType == DataType.SVHN:
        train_data = sio.loadmat('data/train_32x32.mat')
        test_data = sio.loadmat('data/test_32x32.mat')

        X_train = train_data['X']
        y_train = train_data['y'].reshape(-1)
        y_train = np.array(list(map(fix_0, y_train)))
        X_test = test_data['X']
        y_test = test_data['y'].reshape(-1)
        y_test = np.array(list(map(fix_0, y_test)))
        X_train = X_train.reshape(3072, 73257)
        X_test = X_test.reshape(3072, 26032)
        X_train = np.swapaxes(X_train, 0, 1)
        X_test = np.swapaxes(X_test, 0, 1)
        X_train = X_train.reshape(73257, 32, 32, 3)
        X_test = X_test.reshape(26032, 32, 32, 3)
    elif dataType == DataType.CIFAR:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.reshape(50000, 32, 32, 3)
        X_test = X_test.reshape(10000, 32, 32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test
