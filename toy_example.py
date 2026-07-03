import os

import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import kalman_filter


np.random.seed(0)

# Number of time steps
T = 50
dt = 1.0

# State transition matrix
# The state is x_k = [position, velocity]
A = np.array([
    [1, dt],
    [0, 1]
])

# Observation matrix
# We only observe the position
H = np.array([
    [1, 0]
])

# Process noise covariance
Q = np.array([
    [0.01, 0],
    [0, 0.01]
])

# Measurement noise covariance
R = np.array([
    [1.0]
])

# Generate true hidden states
x_true = np.zeros((T, 2))
x_true[0] = np.array([0, 1])

for k in range(1, T):
    q = np.random.multivariate_normal(mean=[0, 0], cov=Q)
    x_true[k] = A @ x_true[k - 1] + q

# Generate noisy measurements
y = np.zeros((T, 1))

for k in range(T):
    r = np.random.normal(loc=0, scale=np.sqrt(R[0, 0]))
    y[k, 0] = (H @ x_true[k])[0] + r

# Initial estimate
x0 = np.array([0, 0])
P0 = np.eye(2) * 10

# Apply Kalman filter
x_filtered, P_filtered = kalman_filter(
    y=y,
    A=A,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0
)

# Create folder for figures
os.makedirs("figures", exist_ok=True)

# Plot position tracking
plt.figure()
plt.plot(x_true[:, 0], label="True position")
plt.scatter(np.arange(T), y[:, 0], label="Noisy measurements", s=15)
plt.plot(x_filtered[:, 0], label="Kalman estimate")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Kalman Filter: Position Tracking")
plt.savefig("figures/position_tracking.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot velocity estimation
plt.figure()
plt.plot(x_true[:, 1], label="True velocity")
plt.plot(x_filtered[:, 1], label="Estimated velocity")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Kalman Filter: Velocity Estimation")
plt.savefig("figures/velocity_estimation.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot estimation error
plt.figure()
plt.plot(np.abs(x_true[:, 0] - x_filtered[:, 0]), label="Position estimation error")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Absolute error")
plt.title("Kalman Filter: Estimation Error")
plt.savefig("figures/estimation_error.png", dpi=300, bbox_inches="tight")
plt.show()
