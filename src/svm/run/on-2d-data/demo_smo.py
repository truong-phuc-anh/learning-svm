from succinctly.datasets import linearly_separable, get_dataset
from svm.algorithms import smo
from svm.algorithms import kernels
from svm.visualize import visualizer
from sklearn import svm

if __name__ == '__main__':
    X_train, y_train = get_dataset(linearly_separable.get_training_examples)
    X_test, y_test = get_dataset(linearly_separable.get_test_examples)
    classifier = smo.SMO(C = 10, tol = 0.001)
    classifier.fit(X_train, y_train)
    b = -classifier.b
    w = classifier.w
    print('w:\n {}'.format(w))
    print('b:\n {}'.format(b))
    #visualizer.visualize_2d_data(X_train, y_train, X_test, y_test, w, b)
    y_pred = classifier.predict(X_test)
    scores = classifier.get_scores(X_test)
    print('scores:\n {}'.format(scores))
    print('y_test:\n {}'.format(y_test))
    print('y_pred:\n {}'.format(y_pred))
    sci = svm.SVC(kernel='linear', C=10, tol=0.001)
    sci.fit(X_train, y_train)
    y_pred = sci.predict(X_test)
    print('y_test:\n {}'.format(y_test))
    print('y_pred:\n {}'.format(y_pred))
