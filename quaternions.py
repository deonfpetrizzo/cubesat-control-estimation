import numpy as np

def quat_mul(q, r):
    q0, qv = q[0], q[1:]
    r0, rv = r[0], r[1:]
    return np.hstack([q0*r0 - np.dot(qv, rv),
                      q0*rv + r0*qv + np.cross(qv, rv)])

def quat_conj(q):
    return np.hstack([q[0], -q[1:]])

def normalize_quat(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12 or np.isnan(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def omega_to_quatdot(q, w):
    q0, qv = q[0], q[1:]
    qdot0 = -0.5 * np.dot(qv, w)
    qdotv = 0.5 * (q0*w + np.cross(qv, w))
    return np.hstack([qdot0, qdotv])

def small_angle_from_quat(q):
    return q[1:]

def quat_to_dcm(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2*q2+q3*q3),   2*(q1*q2+q0*q3),   2*(q1*q3-q0*q2)],
        [2*(q1*q2-q0*q3),     1-2*(q1*q1+q3*q3), 2*(q2*q3+q0*q1)],
        [2*(q1*q3+q0*q2),     2*(q2*q3-q0*q1),   1-2*(q1*q1+q2*q2)]
    ])
