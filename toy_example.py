import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 50
dt = 1.0

A = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]])

Q = np.array([[0.01, 0],
              [0, 0.01]])
R = np.array([[1.0]])

# true state
x_true = np.zeros((T, 2))
x_true[0] = [0, 1]

# measurements
y = np.zeros(T)

for k in range(1, T):
    q = np.random.multivariate_normal([0, 0], Q)
    x_true[k] = A @ x_true[k-1] + q

for k in range(T):
    r = np.random.normal(0, np.sqrt(R[0,0]))
    y[k] = H @ x_true[k] + r

# Kalman filter
m = np.array([0, 0])
P = np.eye(2) * 10

m_hist = np.zeros((T, 2))

for k in range(T):
    # prediction
    if k > 0:
        m_minus = A @ m
        P_minus = A @ P @ A.T + Q
    else:
        m_minus = m
        P_minus = P

    # update
    innovation = y[k] - (H @ m_minus)[0]
    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)

    m = m_minus + (K.flatten() * innovation)
    P = P_minus - K @ S @ K.T

    m_hist[k] = m

plt.figure()
plt.plot(x_true[:, 0], label="True position")
plt.scatter(np.arange(T), y, label="Noisy measurements", s=15)
plt.plot(m_hist[:, 0], label="Kalman estimate")
plt.legend()
plt.xlabel("time")
plt.ylabel("position")
plt.title("Kalman filter: position tracking")
plt.show()

plt.figure()
plt.plot(x_true[:, 1], label="True velocity")
plt.plot(m_hist[:, 1], label="Estimated velocity")
plt.legend()
plt.xlabel("time")
plt.ylabel("velocity")
plt.title("Kalman filter: velocity estimation")
plt.show()
