from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
from Metric import MSE

class GradientDescent:

    def __init__(self, alpha, n_epochs, n_batches, poly=None, lbd=0):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.model = None
        self.n_batches = n_batches
        self.lbd = lbd

        if poly is not None:
            self.poly = PolynomialFeatures(poly)
        else:
            self.poly = None

    def fit(self, x, y):

        if self.poly is not None:
            x = self.poly.fit_transform(x.reshape(-1, 1))
            x = StandardScaler(with_std=True).fit_transform(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples = x.shape[0]
        n_features = x.shape[1]

        idx = np.random.permutation(n_samples)
        x_sh = x[idx]
        y_sh = y[idx]
        lim = int(n_samples*4/5)
        x = x_sh[:lim]
        y = y_sh[:lim]
        x_val = x_sh[lim:]
        y_val = y_sh[lim:]
        n_samples = x.shape[0]

        # 1 - Random w initialization
        w = np.random.random(n_features)

        for epoch in range(self.n_epochs):
            # Shuffle samples and create batches
            batch_size = int(n_samples / self.n_batches)
            idx = np.random.permutation(n_samples)
            x_sh = x[idx]
            y_sh = y[idx]

            for i in range(self.n_batches):
                bx = x_sh[i * batch_size:(i + 1) * batch_size]
                by = y_sh[i * batch_size:(i + 1) * batch_size]
                reg_factor = 1 - 2 * self.lbd * self.alpha
                w = reg_factor * w - self.alpha * (-2 / n_samples) * np.sum((by - bx @ w)[:, np.newaxis] * bx, axis=0)

            mse_error = MSE()
            error_T = mse_error(y_sh, x_sh @ w).round(decimals=2)
            error_V = mse_error(y_val,x_val @ w).round(decimals=2)
            print("Epoch: " + str(epoch) + ", Error de training: " + str(error_T) + ", Error de validacion:" + str(error_V))

        self.model = w

    def predict(self, x):

        if self.poly is not None:
            x = self.poly.fit_transform(x.reshape(-1, 1))
            x = StandardScaler(with_std=True).fit_transform(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return x @ self.model

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.predict(x).reshape(1, -1)