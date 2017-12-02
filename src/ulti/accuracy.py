import numpy as np
def calculate_accuracy(y, y_predit):
    mis_indices = np.where(y != y_predit)[0]
    accuracy = 1.0 - ( 1.0 * len(mis_indices) / y.shape[0])
    return mis_indices, accuracy
