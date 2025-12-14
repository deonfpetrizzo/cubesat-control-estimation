import numpy as np
from geomagnetic import *
from params import *
from quaternions import *

# ============================================================
# Satellite dynamics (true model used for simulation)
# ============================================================
def sat_dynamics(t, x, control_fn):
    q = x[0:4]
    w = x[4:7]
    h = x[7:10]

    Tc, m_dipole = control_fn(q, w, h, t)

    B = geomagnetic_field_body(q, t)
    tau_mag = np.cross(m_dipole, B)

    domega = params.Jinv @ (Tc - tau_mag - np.cross(w, params.J @ w + h))
    dq = omega_to_quatdot(q, w)
    dh = -Tc + tau_mag

    dx = np.zeros(10)
    dx[0:4] = dq
    dx[4:7] = domega
    dx[7:10] = dh
    return dx
