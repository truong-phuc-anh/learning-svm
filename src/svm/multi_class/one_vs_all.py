from succinctly.multi_class import load_X, load_y
import numpy as np
from svm.algorithms import smo

# n_classifiers = n_classes
class OneVsAllClassifier:
    def __init__(self):
        self.classifiers = []
        self.classes = []

    def fit(self, X, y, kernel, C):
        self.classes = np.unique(y)
        y_list = []
        for c in self.classes:
            y_c = np.where(y == c, 1, -1)
            y_list.append(y_c); 
        # Train one binary classifier on each problem (each class vs the rest)
        self.classifiers = []
        for y_i in y_list:
            clf = smo.SMO(kernel=kernel, C=C)
            clf.fit(X, y_i)
            self.classifiers.append(clf)

    def predit(self, X):
        probs = np.zeros((X.shape[0], len(self.classifiers)))
        for idx, clf in enumerate(self.classifiers):
            probs[:, idx] = clf.get_probability_scores(X)
        print('probabilities:\n {}'.format(probs))
        max_indices = np.argmax(probs, axis = 1)
        return [self.classes[idx] for idx in max_indices]
