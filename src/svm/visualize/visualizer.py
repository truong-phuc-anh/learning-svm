import matplotlib.pyplot as plt
import numpy as np

def visualize_2d_data(X_train, y_train, X_test, y_test, w, b):
    fig = plt.figure()

    # setup range
    x = np.arange(0, 12, 0.01)

    # hyperplan formula wx + b = 0
    y = (-w[0]*x - b)/w[1];

    # trainning data
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('trainning set')
    ax.scatter(X_train[:,0], X_train[:,1], c = y_train)
    ax.plot(x, y)

    # test data
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('testing set')
    ax.scatter(X_test[:,0], X_test[:,1], c = y_test)
    ax.plot(x, y)

    plt.show()
