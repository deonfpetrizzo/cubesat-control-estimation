import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from controllers import *
from dynamics import *
from estimation import *
from geomagnetic import *
from quaternions import *
import params

def simulate_bdot_detumble():
    dt = 0.5
    T = 5000
    t_eval = np.arange(0, T + dt, dt)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.deg2rad([0.0, 0.5, 50.0])
    h0 = np.zeros(3)

    # True gyro bias
    b_true = np.deg2rad([0.05, -0.03, 0.02])

    x_true = np.hstack([q0, w0, h0])

    ekf = EKF(dt)
    ekf.x[0:4] = normalize_quat(q0 + np.hstack([0.0, np.deg2rad([0.5, -0.3, 0.2])]))
    ekf.x[4:7] = w0 + np.deg2rad([0.5, -0.3, 0.2])
    ekf.x[7:10] = np.zeros(3)  # initial bias estimate

    x_hist = np.zeros((len(t_eval), 10))
    x_est_hist = np.zeros((len(t_eval), 10))
    m_hist = np.zeros((len(t_eval), 3))

    for i, tt in enumerate(t_eval):
        q_est = ekf.x[0:4]
        w_est = ekf.x[4:7]

        B_est = geomagnetic_field_body(q_est, tt)
        m_cmd = bdot_controller(w_est, B_est)
        m_hist[i, :] = m_cmd

        def ctrl(q, w, h, t):
            return np.zeros(3), m_cmd

        sol_ivp = solve_ivp(
            sat_dynamics,
            [tt, tt + dt],
            x_true,
            args=(ctrl,),
            t_eval=[tt + dt]
        )

        x_true = sol_ivp.y[:, -1]
        x_true[0:4] = normalize_quat(x_true[0:4])
        x_hist[i, :] = x_true.copy()

        q_true = x_true[0:4]
        w_true = x_true[4:7]

        meas = get_measurement_from_truth(q_true, w_true, b_true)

        ekf.predict(Tc=np.zeros(3), m_dipole=m_cmd, t_curr=tt)
        ekf.update(meas)

        x_est_hist[i, :] = ekf.x.copy()

    omega_true = np.rad2deg(x_hist[:, 4:7])
    omega_est = np.rad2deg(x_est_hist[:, 4:7])

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t_eval, omega_true)
    plt.legend(["wx", "wy", "wz"])
    plt.title("True angular rates")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t_eval, omega_est, "--")
    plt.legend(["wx est", "wy est", "wz est"])
    plt.title("EKF estimated angular rates")
    plt.grid()

    plt.tight_layout()
    plt.show()

def simulate_lqr_pointing():
    K = lqr_gain()

    dt = 0.05
    T = 100
    t_eval = np.arange(0, T + dt, dt)

    q0 = normalize_quat(np.array(np.deg2rad([1, 100, 0, 100])))
    w0 = np.deg2rad([2.5, 2, 5])
    h0 = np.zeros(3)

    b_true = np.deg2rad([0.05, -0.03, 0.02])

    x_true = np.hstack([q0, w0, h0])
    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    ekf = EKF(dt)
    ekf.x[0:4] = normalize_quat(q0 + np.hstack([0.0, np.deg2rad([0.5, -0.5, 0.3])]))
    ekf.x[4:7] = w0 + np.deg2rad([0.1, -0.1, 0.2])
    ekf.x[7:10] = np.zeros(3)

    x_hist = np.zeros((len(t_eval), 10))
    x_est_hist = np.zeros((len(t_eval), 10))
    u_hist = np.zeros((len(t_eval), 3))

    for i, tt in enumerate(t_eval):
        q_est = ekf.x[0:4]
        w_est = ekf.x[4:7]

        x_est = np.hstack([q_est[1:], w_est])
        u = -K @ x_est
        u = np.clip(u, -params.rw_torque_max, params.rw_torque_max)
        u_hist[i, :] = u

        def ctrl(q, w, h, t):
            return u, np.zeros(3)

        sol_ivp = solve_ivp(
            sat_dynamics,
            [tt, tt + dt],
            x_true,
            args=(ctrl,),
            t_eval=[tt + dt]
        )

        x_true = sol_ivp.y[:, -1]
        x_true[0:4] = normalize_quat(x_true[0:4])
        x_hist[i, :] = x_true.copy()

        q_true = x_true[0:4]
        w_true = x_true[4:7]

        meas = get_measurement_from_truth(q_true, w_true, b_true)

        ekf.predict(Tc=u, m_dipole=np.zeros(3), t_curr=tt)
        ekf.update(meas)

        x_est_hist[i, :] = ekf.x.copy()

        omega = np.rad2deg(x_hist[:,4:7])
    q_hist = x_hist[:,0:4]

    q_err = np.zeros((len(t_eval),3))
    for i in range(len(t_eval)):
        q_e = quat_mul(quat_conj(q_hist[i,:]), q_des)
        q_err[i,:] = np.rad2deg(q_e[1:])

    q_est_err = np.zeros((len(t_eval),3))
    for i in range(len(t_eval)):
        q_e = quat_mul(quat_conj(x_est_hist[i,0:4]), q_des)
        q_est_err[i,:] = np.rad2deg(q_e[1:])

    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    plt.plot(t_eval, omega[:,0], label='wx true')
    plt.plot(t_eval, omega[:,1], label='wy true')
    plt.plot(t_eval, omega[:,2], label='wz true')
    plt.ylabel("Angular rate (deg/s)")
    plt.title("True angular rates.")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t_eval, x_est_hist[:,4]*180/np.pi, label='wx est')
    plt.plot(t_eval, x_est_hist[:,5]*180/np.pi, label='wy est')
    plt.plot(t_eval, x_est_hist[:,6]*180/np.pi, label='wz est')
    plt.ylabel("Angular rate (deg/s)")
    plt.title("Estimated angular rates (EKF).")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t_eval, q_err[:,0], label='ex true')
    plt.plot(t_eval, q_err[:,1], label='ey true')
    plt.plot(t_eval, q_err[:,2], label='ez true')
    plt.plot(t_eval, q_est_err[:,0], '--', label='ex est')
    plt.plot(t_eval, q_est_err[:,1], '--', label='ey est')
    plt.plot(t_eval, q_est_err[:,2], '--', label='ez est')
    plt.xlabel("Time (s)")
    plt.ylabel("Pointing error (deg)")
    plt.title("Pointing error (orientation) in body axes (true vs EKF estimate).")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def simulate_wheel_desaturation():
    dt = 0.5
    T = 500
    t_eval = np.arange(0, T + dt, dt)

    q0 = normalize_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    w0 = np.zeros(3)
    h0 = np.array([1.6e-3, 1.6e-3, 1.6e-3])  # N*m*s
    x0 = np.hstack([q0, w0, h0])

    m_cmds = np.zeros((len(t_eval), 3))
    h_hist = np.zeros((len(t_eval), 3))
    omega_hist = np.zeros((len(t_eval), 3))

    def ctrl(q, w, h, t):
        B = geomagnetic_field_body(q, t)
        if np.linalg.norm(h) > params.wheel_max_h:
            m = cross_product_controller(h, B, k=1e-2)
        else:
            m = np.zeros(3)
        Tc = np.zeros(3)
        return Tc, m

    sol = []
    x = x0.copy()
    for idx, tt in enumerate(t_eval):
        sol.append(x.copy())
        sol_ivp = solve_ivp(sat_dynamics, [tt, tt+dt], x, args=(ctrl,), t_eval=[tt+dt])
        x = sol_ivp.y[:, -1]
        x[0:4] = normalize_quat(x[0:4])

    sol = np.array(sol)
    for i, tt in enumerate(t_eval):
        xi = sol[i]
        qi = xi[0:4]
        wi = xi[4:7]
        hi = xi[7:10]
        Tc_i, m_i = ctrl(qi, wi, hi, tt)
        m_cmds[i, :] = m_i
        h_hist[i, :] = hi
        omega_hist[i, :] = np.rad2deg(wi)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_eval, h_hist[:,0], label='hx')
    plt.plot(t_eval, h_hist[:,1], label='hy')
    plt.plot(t_eval, h_hist[:,2], label='hz')
    plt.ylabel("Wheel momentum (N*m*s)")
    plt.title("Reaction wheel momentum.")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_eval, m_cmds[:,0], label='mx')
    plt.plot(t_eval, m_cmds[:,1], label='my')
    plt.plot(t_eval, m_cmds[:,2], label='mz')
    plt.ylabel("Magnetorquer dipole (A*m^2)")
    plt.title("Magnetorquer commands.")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t_eval, omega_hist[:,0], label='wx (deg/s)')
    plt.plot(t_eval, omega_hist[:,1], label='wy (deg/s)')
    plt.plot(t_eval, omega_hist[:,2], label='wz (deg/s)')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular rate (deg/s)")
    plt.title("Angular rates during pure desaturation.")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()