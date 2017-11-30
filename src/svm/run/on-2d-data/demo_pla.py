import numpy
import matplotlib.pyplot as plt
from svm.algorithms.pla import *
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

    # evaluation
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X[:,0], X[:,1], c = Y)
    ax.set_title('trainning set')
    x = numpy.arange(0, 15, 0.01)
    y = (-w[1]*x - w[0])/w[2];
    ax.plot(x, y)

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
    
    # evaluation
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(X[:,0], X[:,1], c = Y)
    ax.set_title('testing set')
    x = numpy.arange(0, 15, 0.01)
    y = (-w[1]*x - w[0])/w[2];
    ax.plot(x, y)

    plt.show()


