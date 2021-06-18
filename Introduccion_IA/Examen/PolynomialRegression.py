from BaseModel import BaseModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression


class PolynomialRegression(LinearRegression):

    def __init__(self, grade):
        self.poly = PolynomialFeatures(grade)

    def fit(self, x, y):
        if x.ndim != 1:
            print("Input must have only one feature. Multi-feature is not yet implemented.")
            return
        res = self.poly.fit_transform(x.reshape(-1, 1))
        super().fit(res, y)

    def predict(self, x):
        if x.ndim != 1:
            print("Input must have only one feature. Multi-feature is not yet implemented.")
            return
        return self.poly.fit_transform(x.reshape(-1, 1)) @ self.model
