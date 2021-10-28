import numpy as np

class KalmanFilter(object):
    def __init__(self, r, q, y, x0, p0, a, b, c):
        self.r = r
        self.q = q
        self.y = y
        self.a = a
        #self.b = b
        self.c = c
        self.iter = y.shape[1]
        self.obs = y.shape[0]
        self.x = np.zeros((x0.shape[0], self.iter + 1))
        self.x[:,0] = x0
        self.p = np.zeros((p0.shape[0], p0.shape[1], self.iter + 1))
        self.p[:, :, 0] = p0
        self.kn = np.zeros((p0.shape[0], self.obs, self.iter +1))
        prediction = self._prediction_()

    def _prediction_(self):
        for i in range(1, self.iter + 1):
            x_priori = self.a @ self.x[:, i-1]
            p_priori = self.a @ self.p[:, :, i-1] @ self.a.T + self.q
            self.kn[:, :, i] = (p_priori @ self.c.T) @ np.linalg.inv(self.c @ p_priori @ self.c.T + self.r)
            self.x[:, i] = x_priori + self.kn[:, :, i] @ (self.y[:, i-1] - self.c @ x_priori)
            self.p[:, :, i] = (np.eye(self.p.shape[0]) - self.kn[:, :, i] @ self.c) @ p_priori

    def get_prediction(self):
        return self.x, self.p, self.kn