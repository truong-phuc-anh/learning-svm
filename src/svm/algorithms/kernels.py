import numpy as np

def linear(x1, x2):
    return np.dot(x1, x2)

def polinomial(x1, x2, degree, constrant=0):
    sum = np.dot(x1, x2) + constrant
    return pow(sum, degree)

# radial basis fuction
def rbf(x1, x2, gamma=1e-5):
    return np.exp(-gamma * pow(np.linalg.norm(x1 - x2), 2))
