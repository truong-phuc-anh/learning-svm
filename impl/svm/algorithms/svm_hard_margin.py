import numpy as np
import cvxopt.solvers

# using cvopt solvers to solve quadratics problem of SVMs (calculate alpha(1...m))
# then using alpha(1..m) computer w and b
def svm_hard_margin(X, y):

    # number of samples
    m = X.shape[0] 

    # K: maxtrix (m x m) of all possible dot product of vectors x in X
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

    # get solution: it is a dictionary
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # solution['x'] means get key x in dictionary and it is multiplier in matrix form
    # use np.ravel to convert it to 1D-array
    multipliers = np.ravel(solution['x'])

    # it is a bool 1D array where multiplier > 1e - 7
    has_possitive_multiplier = multipliers > 1e-7
    
    # get support vector (vector x with possitive multiplier)
    # used to calculate b
    support_vectors = X[has_possitive_multiplier]

    # label of those support vector
    # used to calculate b
    suport_vectors_y = y[has_possitive_multiplier]

    # get multipliers of suport vector (used to calculate b)
    sv_multipliers = multipliers[has_possitive_multiplier]

    # calculate w and b
    w = calculate_w(multipliers, X, y)
    b = calculate_b(w, support_vectors, suport_vectors_y)

    return w, b

# calculate w from suport vector infomation (include multipliers of them)
def calculate_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i]
                  for i in range(len(multipliers)))

# calculate b using average method
def calculate_b(w, X, y):
    sum = np.sum(y[i] - np.dot(w, X[i])
                 for i in range(len(X)))
    return sum / len(X)
