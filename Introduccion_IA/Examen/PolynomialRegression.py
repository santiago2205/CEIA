from BaseModel import BaseModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression

class PolynomialRegression(LinearRegression):

    def __init__(self, grade):
        self.poly = PolynomialFeatures(grade)

    def fit(self, x, y):
        res = self.poly.fit_transform(x.reshape(-1, 1))
        super().fit(res, y.reshape(-1,1))

    def predict(self, x):
        return self.poly.fit_transform(x.reshape(-1, 1)) @ self.model

    def fit_transform(self, x, y):
        self.fit(x,y)
        return self.predict(x).reshape(1,-1)