from succinctly.multi_class import load_X, load_y
import numpy as np
from sklearn import svm

# n_classifiers = n_classes
class OneVsAllClassifier:
    def __init__(self):
        self.classifiers = []

    def fit(self, X, y, kernel, C):
        classes = np.unique(y)
        y_list = []
        for c in classes:
            y_c = np.where(y == c, 1, -1)
            y_list.append(y_c);

        # Train one binary classifier on each problem (each class vs the rest)
        self.classifiers = []
        for y_i in y_list:
            clf = svm.SVC(kernel=kernel, C=C)
            clf.fit(X, y_i)
            self.classifiers.append(clf)

    def predit(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for idx, clf in enumerate(self.classifiers):
            predictions[:, idx] = clf.predict(X)
            print('predictions:\n {}'.format(predictions))

        # return the class number if only one classifier predicted it
        # return zero otherwise
        return np.where((predictions == 1).sum(1) == 1,
                        (predictions == 1).argmax(axis=1) + 1,
                        0)




