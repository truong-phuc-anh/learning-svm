from succinctly.datasets import get_dataset, linearly_separable as ls
import numpy as np
import matplotlib.pyplot as plt
from svm.algorithms.svm_hard_margin import *

if __name__ == '__main__':
    X, y = get_dataset(ls.get_training_examples)
    print 'X:\n {0}'.format(X)
    print 'Y:\n{0}'.format(y)
    w, b = svm_hard_margin(X, y)
    print 'w: {0}'.format(w)
    print 'b: {0}'.format(b)

    # evaluation for traning set
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X[:,0], X[:,1], c = y)
    ax.set_title('trainning set')
    x = np.arange(0, 15, 0.01)
    y = (-w[0]*x - b)/w[1];
    ax.plot(x, y)

    X, y = get_dataset(ls.get_test_examples)
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(X[:,0], X[:,1], c = y)
    ax.set_title('testing set')
    x = np.arange(0, 15, 0.01)
    y = (-w[0]*x - b)/w[1];
    ax.plot(x, y)

    plt.show()

