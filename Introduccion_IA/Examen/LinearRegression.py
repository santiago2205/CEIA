from BaseModel import BaseModel
import numpy as np


class LinearRegression(BaseModel):

    def fit(self, x, y):
        if x.ndim == 1:
            w = (x.T @ y) / (x.T @ x)
        else:
            w = np.linalg.inv(x.T @ x) @ x.T @ y
        self.model = w

    def predict(self, x):
        return self.model @ x
