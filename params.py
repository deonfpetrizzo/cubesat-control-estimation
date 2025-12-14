import numpy as np

# ============================================================
# Satellite parameters
# ============================================================
class Params:
    def __init__(self):
        self.Jx = 0.0333
        self.Jy = 0.0333
        self.Jz = 0.0067
        self.J  = np.diag([self.Jx, self.Jy, self.Jz])
        self.Jinv = np.linalg.inv(self.J)

        self.omega_o = 2*np.pi/5400    # orbital rate
        self.max_mag_m = 0.5           # A*m^2
        self.wheel_max_h = 1.5e-3      # N*m*s
        self.rw_torque_max = 1e-3      # N*m

params = Params()
