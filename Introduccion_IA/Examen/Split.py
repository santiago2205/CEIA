import numpy as np

class Split:
    def split_dataset(x, y=None, training_percentage=0.7, validation_percentage=None):
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]
        a = round(x.shape[0] * training_percentage)

        if validation_percentage is None:
            tx = x[:a]
            sx = x[a:]
            if y is None:
                return tx, sx
            else:
                ty = y[:a]
                sy = y[a:]
                return tx, ty, sx, sy
        else:
            b = round(x.shape[0] * (training_percentage + validation_percentage))
            tx = x[:a]
            vx = x[a:b]
            sx = x[b:]
            if y is None:
                return tx, vx, sx
            else:
                ty = y[:a]
                vy = y[a:b]
                sy = y[b:]
                return tx, ty, vx, vy, sx, sy
