import numpy as np

Jx = 0.0333
Jy = 0.0333
Jz = 0.0067
J  = np.diag([Jx, Jy, Jz])
Jinv = np.linalg.inv(J)

omega_o = 2*np.pi/5400    # orbital rate
max_mag_m = 0.5           # A*m^2
wheel_max_h = 1.5e-3      # N*m*s
rw_torque_max = 1e-3      # N*m
