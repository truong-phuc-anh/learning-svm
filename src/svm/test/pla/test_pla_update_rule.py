import numpy
from svm.algorithms.perceptron_learning_algorithm import *

if __name__ == '__main__':
    # testing update rule
    print '------------------------'
    print 'tesing update rule...'
    x = numpy.array([1, 3])
    w = numpy.array([5, 3])
    y_expected = -1
    print 'x = {0}'.format(x)
    print 'w = {0}'.format(w)
    print 'y_expected = {0}'.format(y_expected)

    y_predicted = hypothesis(x, w)
    print 'y_predicted = {0}'.format(y_predicted)

    w = apply_update_rule(w, x, y_expected)
    print 'w_updated = {0}'.format(w)
    y_predicted = hypothesis(x, w)
    print 'y_predicted = {0}'.format(y_predicted)

    w = apply_update_rule(w, x, y_expected)
    print 'w_updated = {0}'.format(w)
    y_predicted = hypothesis(x, w)
    print 'y_predicted = {0}'.format(y_predicted)
