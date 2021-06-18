from BaseModel import BaseModel
import numpy as np


class LinearRegressionB(BaseModel):

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        w = np.linalg.inv(x.T @ x) @ x.T @ y
        self.model = w

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        return x @ self.model
