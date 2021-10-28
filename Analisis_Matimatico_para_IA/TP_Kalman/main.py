import numpy as np
import matplotlib.pyplot as plt
from Dataset import Data
from KalmanFilter import KalmanFilter

# Dataset
dataset = Data('posicion.dat', 'velocidad.dat', 'aceleracion.dat')
position = dataset.get_position()
velocity = dataset.get_velocity()
aceleration = dataset.get_acceleration()

# Condiciones iniciales
x0 = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774])

# Matriz de covarianza de estados iniciales
p0 = np.diag(np.array([100, 100, 100, 1, 1, 1, 0.1, 0.1, 0.1]))

# Matreiz de entrada
B = np.eye(9)

# Tiempo de muestreo
Tk = 1

# Matriz de covarianza A
Id = np.eye(3)
z = np.zeros((3, 3))
A_1 = np.hstack((Id, Id*Tk, Id*((Tk**2)*0.5)))
A_2 = np.hstack(( z,    Id,            Id*Tk))
A_3 = np.hstack(( z,     z,               Id))
A = np.vstack((A_1, A_2, A_3))

# Tomar medidas iniciales
R1, Q1, C1, Y1 = dataset.get_position_gaussian(0, 10)
R2, Q2, C2, Y2 = dataset.get_position_uniform(10)
R3, Q3, C3, Y3 = dataset.get_pos_vel_gaussian(0, 10, 0, 0.2)

# Instanciar las matrices de kalman
kalman_gauss = KalmanFilter(R1, Q1, Y1, x0, p0, A, B, C1)
kalman_uniforme = KalmanFilter(R2, Q2, Y2, x0, p0, A, B, C2)
kalman_pos_vel_gauss = KalmanFilter(R3, Q3, Y3, x0, p0, A, B, C3)

# Tomar predicciones, matrices de covarianza y ganancia de kalman
X1, P1, K1 = kalman_gauss.get_prediction()
X2, P2, K2 = kalman_uniforme.get_prediction()
X3, P3, K3 = kalman_pos_vel_gauss.get_prediction()

# Matriz para ir almacenando los valores dentro del bucle for
buff_position_gaussian = np.zeros((9, 100))
buff_position_uniform = np.zeros((9, 100))
buff_pos_vel_gaussian = np.zeros((9, 100))

for i in range(100):
    R1, Q1, C1, Y1 = dataset.get_position_gaussian(0, 10)
    R2, Q2, C2, Y2 = dataset.get_position_uniform(10)
    R3, Q3, C3, Y3 = dataset.get_pos_vel_gaussian(0, 10, 0, 0.2)
    kalman_gauss = KalmanFilter(R1, Q1, Y1, x0, p0, A, B, C1)
    kalman_uniforme = KalmanFilter(R2, Q2, Y2, x0, p0, A, B, C2)
    kalman_pos_vel_gauss = KalmanFilter(R3, Q3, Y3, x0, p0, A, B, C3)
    X1, P1, K1 = kalman_gauss.get_prediction()
    X2, P2, K2 = kalman_uniforme.get_prediction()
    X3, P3, K3 = kalman_pos_vel_gauss.get_prediction()
print(Y1.shape)
time = np.arange(Y1.shape[1])*Tk
plt.figure(1)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[0, 1:], label="Prediccion Gaussiana, Pos X")
plt.plot(time, X2[0, 1:], label="Prediccion Uniforme, Pos X")
plt.plot(time, X3[0, 1:], label="Prediccion Vel-Pos Gaussiana, Pos X")
plt.plot(time, position['var_x'], label="Posicion verdadera en X")
plt.plot(time, Y1[0, :], label="Posicion medida en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[1, 1:], label="Prediccion Gaussiana, Pos Y")
plt.plot(time, X2[1, 1:], label="Prediccion Uniforme, Pos Y")
plt.plot(time, X3[1, 1:], label="Prediccion Vel-Pos Gaussiana, Pos Y")
plt.plot(time, position['var_y'], label="Posicion verdadera en Y")
plt.plot(time, Y1[1, :], label="Posicion medida en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[2, 1:], label="Prediccion Gaussiana, Pos Z")
plt.plot(time, X2[2, 1:], label="Prediccion Uniforme, Pos Z")
plt.plot(time, X3[2, 1:], label="Prediccion Vel-Pos Gaussiana, Pos Z")
plt.plot(time, position['var_z'], label="Posicion verdadera en Z")
plt.plot(time, Y1[2, :], label="Posicion medida en Z")
plt.legend()

plt.figure(2)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[3, 1:], label="Prediccion Gaussiana, Vel X")
plt.plot(time, X2[3, 1:], label="Prediccion Uniforme, Vel X")
plt.plot(time, X3[3, 1:], label="Prediccion Vel-Pos Gaussiana, Vel X")
plt.plot(time, velocity['var_x'], label="Velocidad verdadera en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[4, 1:], label="Prediccion Gaussiana, Vel Y")
plt.plot(time, X2[4, 1:], label="Prediccion Uniforme, Vel Y")
plt.plot(time, X3[4, 1:], label="Prediccion Vel-Pos Gaussiana, Vel Y")
plt.plot(time, velocity['var_y'], label="Velocidad verdadera en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[5, 1:], label="Prediccion Gaussiana, Vel Z")
plt.plot(time, X2[5, 1:], label="Prediccion Uniforme, Vel Z")
plt.plot(time, X3[5, 1:], label="Prediccion Vel-Pos Gaussiana, Vel Z")
plt.plot(time, velocity['var_z'], label="Velocidad verdadera en Z")
plt.legend()

plt.figure(3)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[6, 1:], label="Prediccion Gaussiana, Acel X")
plt.plot(time, X2[6, 1:], label="Prediccion Uniforme, Acel X")
plt.plot(time, X3[6, 1:], label="Prediccion Vel-Pos Gaussiana, Acel X")
plt.plot(time, aceleration['var_x'], label="Aceleracion verdadera en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[7, 1:], label="Prediccion Gaussiana, Acel Y")
plt.plot(time, X2[7, 1:], label="Prediccion Uniforme, Acel Y")
plt.plot(time, X3[7, 1:], label="Prediccion Vel-Pos Gaussiana, Acel Y")
plt.plot(time, aceleration['var_y'], label="Aceleracion verdadera en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacio entre los diferentes filtros de prediccion vs los valores verdaderos')
plt.plot(time, X1[8, 1:], label="Prediccion Gaussiana, Acel Z")
plt.plot(time, X2[8, 1:], label="Prediccion Uniforme, Acel Z")
plt.plot(time, X3[8, 1:], label="Prediccion Vel-Pos Gaussiana, Acel Z")
plt.plot(time, aceleration['var_z'], label="Aceleracion verdadera en Z")
plt.legend()

plt.figure(4)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X1[0, 1:], label="Prediccion Gaussiana en X")
plt.plot(time, Y1[0, :], label="Posicion medida Gaussiana en X")
plt.plot(time, position['var_x'], label="Posicion verdadera en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X1[1, 1:], label="Prediccion Gaussiana en Y")
plt.plot(time, Y1[1, :], label="Posicion medida Gaussiana en Y")
plt.plot(time, position['var_y'], label="Posicion verdadera en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X1[2, 1:], label="Prediccion Gaussiana en Z")
plt.plot(time, Y1[2, :], label="Posicion medida Gaussiana en Z")
plt.plot(time, position['var_z'], label="Posicion verdadera en Z")
plt.legend()

plt.figure(5)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X2[0, 1:], label="Prediccion Uniforme en X")
plt.plot(time, Y2[0, :], label="Posicion medida Uniforme en X")
plt.plot(time, position['var_x'], label="Posicion verdadera en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X2[1, 1:], label="Prediccion Uniforme en Y")
plt.plot(time, Y2[1, :], label="Posicion medida Uniforme en Y")
plt.plot(time, position['var_y'], label="Posicion verdadera en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X2[2, 1:], label="Prediccion Uniforme en Z")
plt.plot(time, Y2[2, :], label="Posicion medida Uniforme en Z")
plt.plot(time, position['var_z'], label="Posicion verdadera en Z")
plt.legend()

plt.figure(6)
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X3[0, 1:], label="Prediccion Pos-Vel Gaussiana en X")
plt.plot(time, Y3[0, :], label="Posicion medida Pos-Vel Gaussiana en X")
plt.plot(time, position['var_x'], label="Posicion verdadera en X")
plt.legend()

plt.subplot(312)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X3[1, 1:], label="Prediccion Pos-Vel Gaussiana en Y")
plt.plot(time, Y3[1, :], label="Posicion medida Pos-Vel Gaussiana en Y")
plt.plot(time, position['var_y'], label="Posicion verdadera en Y")
plt.legend()

plt.subplot(313)
plt.grid(True)
plt.title('Comparacion entre los valores estimados, medidos y reales')
plt.plot(time, X3[2, 1:], label="Prediccion Pos-Vel Gaussiana en Z")
plt.plot(time, Y3[2, :], label="Posicion medida Pos-Vel Gaussiana en Z")
plt.plot(time, position['var_z'], label="Posicion verdadera en Z")
plt.legend()

plt.show()
