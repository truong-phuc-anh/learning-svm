from svm.multi_class.one_vs_one import *
from succinctly.multi_class import load_X, load_y
from svm.algorithms import kernels

if __name__ == '__main__':
    X = load_X()
    y = load_y()
    print('X:\n {}'.format(X))
    print('y:\n {}'.format(y))
    multi_classifier = OneVsOneClassifier()
    multi_classifier.fit(X, y, kernel = kernels.linear, C = 1000)
    X_unknown = np.array([[5, 5], [2, 5], [-10, -100]])
    predict_y = multi_classifier.predit(X_unknown)
    print('y_pred:\n {}'.format(predict_y))