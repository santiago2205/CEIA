from Dataset import Dataset
from Split import Split
from PolynomialRegression import PolynomialRegression
from Metric import MSE
from MiniBatch import GradientDescent
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


def k_folds(X_train, y_train, grado, k=5):
    # Definimos el modelo a emplear
    p_regression = PolynomialRegression(grado)

    # Definimos una métrica
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []

    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        p_regression.fit(new_X_train, new_y_train.reshape(-1, 1))
        prediction = p_regression.predict(new_X_valid)
        k_error = error(new_y_valid.reshape(-1,1), prediction)
        mse_list.append(k_error)

    mean_mse = np.mean(mse_list)

    return mean_mse



# Importamos el dataset y hacemos el split
ds = Dataset('data/clase_8_dataset')
X = ds.data['entrada']
y = ds.data['salida']
X_norm = StandardScaler(with_std=True).fit_transform(X.reshape(-1, 1))
plt.scatter(X, y)
plt.show()

X_train, y_train, X_test, y_test = Split.split_dataset(X_norm, y, 0.8)

# K-Folds para mejor aproximacion
lr_list = np.array([1, 2, 3, 4])
kfolds_lr = np.zeros(lr_list.shape)
print("Entre")
for i in range (lr_list.shape[0]):
    kfolds_lr[i] = k_folds(X_train, y_train, i+1)
    print("{:.10f}".format(kfolds_lr[i]))

plt.plot(lr_list, kfolds_lr)
plt.show()

#Por sucesivas iteracionas, se observa que el grado polinomico que tiene menor error es el 3

p_regression = PolynomialRegression(3)
y_pol_train = p_regression.fit_transform(X_train, y_train)
y_pol_test = p_regression.predict(X_test)

x_cl = p_regression.model
x = np.linspace(X_test.min(), X_test.max(), 100)
y_cl = x_cl[0] * x**0 + x_cl[1] * x**1 + x_cl[2] * x**2 + x_cl[3] * x**3

plt.plot(x,y_cl)
plt.scatter(X_test, y_test)
plt.show()

# Ahora aplicamos MiniBatch Gradiente Descendiente
MB = GradientDescent(alpha=0.1, n_epochs=50, n_batches=15, poly=3, lbd=0.001)
MB.fit(X_train, y_train)

y_pred = MB.predict(X_test)
#y_pred = y_pred + (y_test-y_pred) #Solventando error de bias
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, color='red')
plt.show()

# Comparamos el modelo obtenido de MiniBatch y el polinomico

x_mb = MB.model
x = np.linspace(X_test.min(), X_test.max(), 100)
y_mb = x_mb[0] * x**0 + x_mb[1] * x**1 + x_mb[2] * x**2 + x_mb[3] * x**3

plt.plot(x,y_cl, color='red')
plt.plot(x,y_mb)
plt.show()