from succinctly.datasets import get_dataset, linearly_separable as ls
import cvxopt.solvers
import numpy as np
from svm.algorithms.svm_hard_margin import *

if __name__ == '__main__':

    X, y = get_dataset(ls.get_training_examples)
    m = X.shape[0]
    print 'X:\n {0}'.format(X)
    print 'Y:\n{0}'.format(y)
    print 'number of samples:\n {0}'.format(m)

    K = np.array([np.dot(X[i], X[j])
                 for i in range(m)
                 for j in range(m)]).reshape(m, m)

    # P: matrix m x m parameter for cvxopt solver
    # np.outer: calculate outer product of y and y-> return a matrix m x m
    # example outer product between 2 vector https://www.google.com.vn/search?q=outer+product&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiHp7b0kNnXAhUFFpQKHXPOAKwQ_AUICigB&biw=1366&bih=588#imgrc=RjQDwZ3UhQ5cLM:
    P = cvxopt.matrix(np.outer(y, y) * K)

    # q: 1 x m matrix parameter for cvopt solver
    q = cvxopt.matrix(-1 * np.ones(m))
    
    # G: m x m zero matrix with -1 on diagonal
    # example: m = 3, G =
    # -1  0  0
    #  0 -1  0
    #  0  0 -1
    G = cvxopt.matrix(np.diag(-1 * np.ones(m)))

    # h: 1 x m zero matrix
    h = cvxopt.matrix(np.zeros(m))
    
    # A: 1 x m
    A = cvxopt.matrix(y, (1, m))

    # b: 1 x 1
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    multipliers = np.ravel(solution['x'])
    print 'solution[\'x\']:\n {0}'.format(solution['x'])
    print 'multipliers:\n {0}'.format(multipliers)

    has_possitive_multiplier = multipliers > 1e-7
    print 'has_possitive_multiplier:\n {0}'.format(has_possitive_multiplier)

    sv_multipliers = multipliers[has_possitive_multiplier]
    print 'sv_multipliers:\n {0}'.format(sv_multipliers)

    # get support vector (vector x with possitive multiplier)
    # used to calculate w and b
    support_vectors = X[has_possitive_multiplier]
    print 'support vectors:\n {0}'.format(support_vectors)

    # label of those support vector
    suport_vectors_y = y[has_possitive_multiplier]
    print 'suport_vectors_y:\n {0}'.format(suport_vectors_y)

    # calculate w and b
    w = calculate_w(sv_multipliers, support_vectors, suport_vectors_y)
    b = calculate_b(w, support_vectors, suport_vectors_y)

    print ('w: {0}').format(w)
    print ('b: {0}').format(b)

