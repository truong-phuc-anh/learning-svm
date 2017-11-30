import numpy
from svm.algorithms.perceptron_learning_algorithm import *

if __name__ == '__main__':
    X = numpy.array([[-1, -1], [2, 2], [3, 3], [-4, -4], [-5, -5]])
    Y = numpy.array([1, 1, 1, 1, 1])
    w = numpy.array([1, 1])
    mis_samples = predict(hypothesis, X, Y, w)
    print 'mis_samples = {0}'.format(mis_samples)
    a_sample, y_expected = pick_one_from(mis_samples, X, Y)
    print 'pick_one_from = {0}, {1}'.format(a_sample, y_expected)
