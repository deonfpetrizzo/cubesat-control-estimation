import numpy as np
from geomagnetic import *
from quaternions import *
from params import *
# ============================================================
# Extended Kalman Filter (quaternion + angular rate)
# ============================================================
class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(7) 
        self.x[0] = 1.0  # initial quaternion = identity
        self.P = np.eye(7) * 1e-3 

        # process noise
        q_att = 1e-8
        q_omega = 1e-6
        self.Q = np.diag([q_att, q_att, q_att, q_att, q_omega, q_omega, q_omega])

        # measurement noise (star tracker + gyro)
        # star tracker quaternion (4) low noise; gyro higher noise
        self.R = np.diag([2e-3, 2e-3, 2e-3, 2e-3,   # quaternion
                          1e-2, 1e-2, 1e-2])        # gyro

    def predict(self, Tc, m_dipole, t_curr):
        """
        Propagate state forward dt using a simple RK4 integration of the process model.
        We need B-field to compute tau_mag so we pass current time t_curr and use q from state.
        Approximate omega dynamics without wheel momentum h (estimator simplification).
        """
        def f(state):
            q = state[0:4]
            w = state[4:7]
            B = geomagnetic_field_body(q, t_curr)
            tau_mag = np.cross(m_dipole, B)
            domega = params.Jinv @ (Tc - tau_mag - np.cross(w, params.J @ w))
            dq = omega_to_quatdot(q, w)
            return np.hstack([dq, domega])

        # RK4 integration for the propagation
        x0 = self.x.copy()
        k1 = f(x0)
        k2 = f(x0 + 0.5*self.dt*k1)
        k3 = f(x0 + 0.5*self.dt*k2)
        k4 = f(x0 + self.dt*k3)
        x_pred = x0 + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_pred[0:4] = normalize_quat(x_pred[0:4])

        # Covariance propagation using numerical jacobian
        F = self._numerical_jacobian_state(f, x0)
        Fd = np.eye(7) + F*self.dt
        self.P = Fd @ self.P @ Fd.T + self.Q
        self.x = x_pred

    def update(self, y):
        """
        y contains [q_meas(4), w_meas(3)]
        measurement model h(x) = [q, w]
        """
        h_pred = self.x.copy() 
        H = np.eye(7)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y_vec = np.hstack([y[0:4], y[4:7]])
        dx = K @ (y_vec - h_pred)
        self.x = self.x + dx
        self.x[0:4] = normalize_quat(self.x[0:4])
        self.P = (np.eye(7) - K @ H) @ self.P

    def _numerical_jacobian_state(self, f, x, eps=1e-6):
        n = len(x)
        F = np.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            fp = f(xp)
            F[:, i] = (fp - f0) / eps
        return F


# ============================================================
# Sensor models and measurement noise updates
# ============================================================
def noisy_star_tracker(q_true, sigma_vec=2e-3):
    """ Simulate star tracker quaternion measurement by adding small noise to the quaternion vector part. """
    q = q_true.copy()
    q_noisy = q.copy()
    q_noisy[1:] += np.random.randn(3) * sigma_vec
    q_noisy = normalize_quat(q_noisy)
    return q_noisy

def noisy_gyro(w_true, sigma_w=1e-2):
    """ Simulate gyro angular rate measurement by adding small noise to each coordinate. """
    return w_true + np.random.randn(3)*sigma_w

def get_measurement_from_truth(q_true, w_true):
    """ Combine star tracker + gyro measurements. """
    q_meas = noisy_star_tracker(q_true)  
    w_meas = noisy_gyro(w_true)
    return np.hstack([q_meas, w_meas])