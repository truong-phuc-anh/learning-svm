import numpy
from svm.algorithms.perceptron_learning_algorithm import *
from succinctly.datasets import get_dataset, linearly_separable

if __name__ == '__main__':
    numpy.random.seed(2017)
    print '------trainning-------'
    X, Y = get_dataset(linearly_separable.get_training_examples)
    # add 1 to all x in X to get augmented vectors
    X_augmented = numpy.c_[numpy.ones(X.shape[0]), X]
    print 'training set:\n {0}'.format(X_augmented)
    print 'expected labels:\n {0}'.format(Y)
    w = pla(X_augmented, Y) # different w with different runing time (change random seed will see)
    print w

    print '------testing-------'
    X, Y = get_dataset(linearly_separable.get_test_examples)
    # add 1 to all x in X to get augmented vectors
    X_augmented = numpy.c_[numpy.ones(X.shape[0]), X]
    print 'testing set:\n {0}'.format(X_augmented)
    print 'expected labels:\n {0}'.format(Y)
    Y_predicted = numpy.apply_along_axis(hypothesis, 1, X_augmented, w)
    print 'predicted labels:\n {0}'.format(Y_predicted)

    mis_samples = predict(hypothesis, X_augmented, Y, w)
    print 'number of mis_samples = {0}'.format(len(mis_samples))
    print 'mis_samples percent = {0} %'.format(len(mis_samples) * 100.0 / len(Y_predicted))
    
