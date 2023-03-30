#from math import *
import numpy as np

#norm = lambda v: sqrt(sum(v**2))

#norm = lambda v: (sum(v**2))**0.5

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def costos(X, y, t): # Método para calcular los costos del modelo.
    
    #costs = []

    m = y.size 

    # print("len(X): ", len(X))
    # print("len(y): ", len(y))
    # print("len(t): ", len(t))

    h = sigmoid(X @ t)
    
    # print("H: ", h)
    # print("Y: ", y)

    # Calculando el J.
    #J = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))

    grad = (1/m) * (X.T @ (h - y))

    # print("H: ", h)
    # print("J: ", J)
    # print("grad: ", grad)
    
    return grad

def gradiente(X, y, t, a=0.1, n=10000): # Método para calcular el descenso del gradiente.

    for i in range(n):
        grad = costos(X, y, t)
        t -= a * grad
    
    # Haciendo una predicción.
    predictions = sigmoid(X @ t)

    # Ajustando la predicción a su etiqueta binaria.
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    return grad, t, predictions

def poly_features(X, degree):
    return np.vstack([np.power(X, i) for i in range(1, degree+1)]).T

# Definir una función para ajustar un modelo de regresión polinómica de un grado específico
def poly_fit(X, y, degree):
    X_poly = poly_features(X, degree)
    theta = np.linalg.inv(X_poly.T @ X_poly) @ (X_poly.T @ y)
    return theta

# Definir una función para calcular el costo de un modelo de regresión polinómica de un grado específico
def poly_cost(X, y, theta):
    m = y.size
    X_poly = poly_features(X, theta.size)
    J = (1/(2*m)) * np.sum(np.power(X_poly @ theta - y, 2))
    return J
