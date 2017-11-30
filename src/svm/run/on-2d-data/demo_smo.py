from succinctly.datasets import linearly_separable, get_dataset
from svm.algorithms.smo_algorithm import SmoAlgorithm
from svm.algorithms.kernels import *
from svm.algorithms.svm_hard_margin import *
from svm.visualize.visualizer import *

if __name__ == '__main__':
    X_train, y_train = get_dataset(linearly_separable.get_training_examples)
    X_test, y_test = get_dataset(linearly_separable.get_test_examples)
    smo = SmoAlgorithm(X_train, y_train, C = 10, tol = 0.001, kernel = linear_kernel, use_linear_optim = True)
    smo.main_routine()
    b = -smo.b # smoalgorithm implement with hyperplan wx - b = 0, and we use wx + b =0
    w = calculate_w(smo.alphas, X_train, y_train) # calculate w  with the same formula with hard margin
    margin = calculate_margin(w)
    print('w:\n {}'.format(w))
    print('b:\n {}'.format(b))
    print('margin:\n {}'.format(margin))
    visualize_2d_data(X_train, y_train, X_test, y_test, w, b)
