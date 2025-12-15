import numpy as np
from geomagnetic import *
from quaternions import *
import params


class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(10)
        self.x[0] = 1.0 

        self.P = np.eye(10) * 1e-3

        q_att = 1e-8            # attitude noise
        q_omega = 1e-6          # angular acceleration noise
        q_bias = 1e-10          # gyro bias random walk

        self.Q = np.diag([
            q_att, q_att, q_att, q_att,       # quaternion
            q_omega, q_omega, q_omega,        # omega
            q_bias, q_bias, q_bias            # bias
        ])

        sigma_q = 2e-3 # star tracker noise
        sigma_w = 1e-2 # gyro noise (assume order of magnitued noiser than star tracker)

        self.R = np.diag([
            sigma_q, sigma_q, sigma_q, sigma_q,
            sigma_w, sigma_w, sigma_w
        ])


    def predict(self, Tc, m_dipole, t_curr):
        def f(state):
            q = state[0:4]
            w = state[4:7]
            b = state[7:10]

            B = geomagnetic_field_body(q, t_curr)
            tau_mag = np.cross(m_dipole, B)

            domega = params.Jinv @ (Tc - tau_mag - np.cross(w, params.J @ w))
            dq = omega_to_quatdot(q, w)

            db = np.zeros(3)  # bias random walk (mean zero)

            return np.hstack([dq, domega, db])

        # RK4 integration
        x0 = self.x.copy()
        k1 = f(x0)
        k2 = f(x0 + 0.5*self.dt*k1)
        k3 = f(x0 + 0.5*self.dt*k2)
        k4 = f(x0 + self.dt*k3)

        x_pred = x0 + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_pred[0:4] = normalize_quat(x_pred[0:4])

        # Covariance propagation
        F = self._numerical_jacobian_state(f, x0)
        Fd = np.eye(10) + F*self.dt
        self.P = Fd @ self.P @ Fd.T + self.Q

        self.x = x_pred


    def update(self, y):
        q = self.x[0:4]
        w = self.x[4:7]
        b = self.x[7:10]

        h_pred = np.hstack([
            q,
            w - b
        ])

        # Measurement Jacobian
        H = np.zeros((7, 10))
        H[0:4, 0:4] = np.eye(4)      
        H[4:7, 4:7] = np.eye(3)      
        H[4:7, 7:10] = -np.eye(3)  

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ (y - h_pred)
        self.x += dx
        self.x[0:4] = normalize_quat(self.x[0:4])

        self.P = (np.eye(10) - K @ H) @ self.P

    def _numerical_jacobian_state(self, f, x, eps=1e-6):
        n = len(x)
        F = np.zeros((n, n))
        f0 = f(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            fp = f(xp)
            F[:, i] = (fp - f0) / eps
        return F

def noisy_star_tracker(q_true, sigma_vec=2e-3):
    q = q_true.copy()
    q_noisy = q.copy()
    q_noisy[1:] += np.random.randn(3)*sigma_vec
    q_noisy = normalize_quat(q_noisy)
    return q_noisy

def noisy_gyro(w_true, b_true, sigma_vec=1e-4):
    return w_true - b_true - np.random.randn(3)*sigma_vec

def get_measurement_from_truth(q_true, w_true, b_true):
    q_meas = noisy_star_tracker(q_true)
    w_meas = noisy_gyro(w_true, b_true)
    return np.hstack([q_meas, w_meas])
