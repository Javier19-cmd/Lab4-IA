import numpy as np

sigmoid = lambda X, t: 1 / (1 + np.exp(-(X @ t)))
cost = lambda X, y, t: ((sigmoid(X, t) - y) ** 2).sum() / len(y)
grad = lambda X, y, t: 2 * X.T @ (X @ t - y) / len(y)
