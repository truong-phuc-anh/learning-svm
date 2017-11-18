import numpy
from svm.algorithms.perceptron_learning_algorithm import *

if __name__ == '__main__':
    numpy.random.seed(2017)

    x = numpy.random.rand(default_dimension)
    print 'x = {0}'.format(x)
    
    w = numpy.random.rand(default_dimension);
    print 'w = {0}'.format(w)

    y_predicted = hypothesis(x, w)
    print 'predict label of x: {0}'.format(y_predicted)
