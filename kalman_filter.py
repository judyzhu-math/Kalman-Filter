import numpy as np


def kalman_filter(y, A, H, Q, R, x0, P0):
    """
    Kalman filter for a linear Gaussian state-space model.

    Model:
        x_k = A x_{k-1} + q_{k-1},      q ~ N(0, Q)
        y_k = H x_k + r_k,              r ~ N(0, R)

    Parameters
    ----------
    y : array-like
        Observations with shape (n_steps, observation_dim).
    A : array-like
        State transition matrix with shape (state_dim, state_dim).
    H : array-like
        Observation matrix with shape (observation_dim, state_dim).
    Q : array-like
        Process noise covariance matrix with shape (state_dim, state_dim).
    R : array-like
        Measurement noise covariance matrix with shape (observation_dim, observation_dim).
    x0 : array-like
        Initial state estimate with shape (state_dim,).
    P0 : array-like
        Initial covariance matrix with shape (state_dim, state_dim).

    Returns
    -------
    x_filtered : ndarray
        Filtered state estimates with shape (n_steps, state_dim).
    P_filtered : ndarray
        Filtered covariance matrices with shape (n_steps, state_dim, state_dim).
    """

    y = np.asarray(y)
    A = np.asarray(A)
    H = np.asarray(H)
    Q = np.asarray(Q)
    R = np.asarray(R)
    x = np.asarray(x0)
    P = np.asarray(P0)

    n_steps = y.shape[0]
    state_dim = x.shape[0]

    x_filtered = np.zeros((n_steps, state_dim))
    P_filtered = np.zeros((n_steps, state_dim, state_dim))

    I = np.eye(state_dim)

    for k in range(n_steps):
        # Prediction step
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # Innovation
        innovation = y[k] - H @ x_pred
        innovation_covariance = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(innovation_covariance)

        # Update step
        x = x_pred + K @ innovation
        P = (I - K @ H) @ P_pred

        x_filtered[k] = x
        P_filtered[k] = P

    return x_filtered, P_filtered
