import numpy as np
from scipy.linalg import solve_continuous_are
from geomagnetic import *
from params import *
from quaternions import *

# ============================================================
# Controllers
# ============================================================
def bdot_controller(omega, B, k=1.2e4):
    m = -k*np.cross(omega, B)
    n = np.linalg.norm(m)
    if n > params.max_mag_m:
        m *= params.max_mag_m / n
    return m

def lqr_gain():
    Jx, Jy, Jz = params.Jx, params.Jy, params.Jz
    wo = params.omega_o

    A = np.zeros((6,6))
    A[0,3] = 0.5
    A[1,4] = 0.5
    A[2,5] = 0.5

    A[3,0] = 8*(Jz-Jy) * wo**2 - 2*wo
    A[3,5] = (-Jx-Jz+Jy) * wo
    A[4,1] = 6*(Jz-Jx) * wo**2
    A[5,2] = 2*(Jx-Jy) * wo**2 - 2*wo
    A[5,3] = -A[3,5]

    B = np.zeros((6,3))
    B[3:,:] = params.Jinv

    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1, 1, 1])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def cross_product_controller(h, B, k=15000.0, h_ref=None):
    if h_ref is None:
        h_ref = np.zeros(3)
    he = h - h_ref

    Bn2 = np.dot(B, B)
    if Bn2 < 1e-12:
        return np.zeros(3)

    m = -(k/Bn2)*np.cross(B, he)

    n = np.linalg.norm(m)
    if n > params.max_mag_m:
        m *= params.max_mag_m / n
    return m