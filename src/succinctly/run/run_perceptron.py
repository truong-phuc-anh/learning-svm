import numpy as np
from succinctly.datasets import get_dataset, linearly_separable as ls
from succinctly.algorithms.pla import pla

if __name__ == '__main__':
    np.random.seed(88)

    X, y = get_dataset(ls.get_training_examples)

    # transform X into an array of augmented vectors.
    X_augmented = np.c_[np.ones(X.shape[0]), X]

    w = pla(X_augmented, y)

    print(w) # [-44.35244895   1.50714969   5.52834138]


