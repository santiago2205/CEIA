from Metric import Metric
import numpy as np


class MSE(Metric):
    def __call__(self, target, prediction):
        return np.sum((target - prediction) ** 2) / target.shape[0]
