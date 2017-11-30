import numpy
from svm.algorithms.perceptron_learning_algorithm import *

if __name__ == '__main__':
    print 'Test predict() in pla'
    X = numpy.array([[1, 1], [-1, -1], [-1, -2], [1, 5]])
    Y_expected = numpy.array([1, 1, 1, 1])
    w = numpy.array([1, 1])
    mis_samples = predict(hypothesis, X, Y_expected, w)
    print mis_samples
