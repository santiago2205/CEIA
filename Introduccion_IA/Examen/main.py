from Dataset import Dataset
from Split import Split
from PolynomialRegression import PolynomialRegression
from Metric import Accuracy
from Metric import MSE
import matplotlib.pyplot as plt
import numpy as np


def k_folds(X_train, y_train, grado, k=5):
    # Definimos el modelo a emplear
    p_regression = PolynomialRegression(grado)

    # Definimos una métrica
    error = Accuracy()

    chunk_size = int(len(X_train) / k)
    acc_list = []

    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        p_regression.fit(new_X_train, new_y_train.reshape(-1, 1))
        prediction = p_regression.predict(new_X_valid)
        k_error = error(new_y_valid, prediction)
        acc_list.append(k_error)
        # print("Modelo {i} de {k}, Métrica: {error}".format(i=i/chunk_size, k=k, error=k_error))

    mean_acc = np.mean(acc_list)

    return mean_acc



# Importamos el dataset y hacemos el split
ds = Dataset('data/clase_8_dataset')
X_train, y_train, X_test, y_test = Split.split_dataset(ds.data['entrada'], ds.data['salida'], 0.8)
# plt.scatter(X_train, y_train)
# plt.show()

# K-Folds para mejor aproximacion
lr_list = np.array([1, 2, 3, 4])
kfolds_lr = np.zeros(lr_list.shape)
print("Entre")
for i in range (lr_list.shape[0]):
    kfolds_lr[i] = k_folds(X_train, y_train.reshape(-1, 1), i+1)
    print("{:.10f}".format(kfolds_lr[i]))

best_lr = lr_list[np.argmax(kfolds_lr)]
print("Lauti doy lastima")
print(best_lr)

