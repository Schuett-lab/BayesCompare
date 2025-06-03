import numpy as np
from scipy.linalg import pinv

alpha = 0
prec_b = 1 / (1 - alpha)
if alpha == 0:
    prec_n = np.inf
else:
    prec_n = 1 / alpha

n_dim = 20
N = 10
n_new = 100

X = np.random.rand(N, n_dim)
y = np.random.rand(N) + X[:, 0] - 0.5


X_new = np.random.rand(n_new, n_dim)
if alpha < 1 and alpha > 0:
    # normal way including theta:
    theta = prec_n * pinv(prec_n * X.T @ X + prec_b * np.eye(n_dim)) @ X.T @ y  # type: ignore
    y_new = X_new @ theta
    # direct formula putting theta in
    y_new2 = X_new @ \
        np.linalg.inv(X.T @ X * prec_n / prec_b + np.eye(n_dim)) * \
        prec_n / prec_b @ X.T @ y
    # matrix inversion formula
    y_new3 = X_new @ (np.eye(n_dim) - X.T @ np.linalg.inv(np.eye(N) + X@X.T * prec_n / prec_b)
                      @ X * prec_n / prec_b) * prec_n / prec_b @ X.T @ y
    # back to alpha parametrization
    y_new4 = X_new @ (
        np.eye(n_dim)
        - X.T @ np.linalg.inv(alpha / (1-alpha) * np.eye(N) + X@X.T) @ X
        ) * (1 - alpha) / alpha @ X.T @ y
    # to kernel formulation
    y_new5 = (
        X_new @ X.T
        - X_new @ X.T @ np.linalg.inv(alpha / (1-alpha) * np.eye(N) + X@X.T) @ X @ X.T
        ) * (1 - alpha) / alpha  @ y
elif alpha == 0:  # no noise
    # normal way including theta:
    theta = pinv(X.T @ X) @ X.T @ y
    y_new = X_new @ theta
    # direct formula putting theta in
    y_new2 = X_new @ pinv(X.T @ X) @ X.T @ y
    y_new2a = X_new @ pinv(X) @ y
    y_new2b = X_new @ pinv(X.T @ X) @ pinv(X.T @ X).T @ X.T @ (X @ X.T) @ y
    # murphy kernel trick (17.108)
    y_new3 = X_new @ X.T @ pinv(X @ X.T) @ y
elif alpha == 1:  # only prior
    y_new = np.zeros(n_dim)
else:
    raise ValueError("alpha must be between 0 and 1")
