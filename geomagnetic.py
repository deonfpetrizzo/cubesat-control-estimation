import numpy as np
from quaternions import *


# Geomagnetic field model (Tilted Dipole, ECI to ECEF to Body)
mu_earth = 3.986004418e14
Re = 6371e3
wE = 7.292115e-5        # Earth rotation rate
M_EARTH = 7.94e22       # Dipole moment
incl = np.deg2rad(51.6) # Orbital inclination for simulation

def orbit_position_ecef(t, a=6771e3, inc=incl):
    """ Returns satellite inertial position in ECEF (m). """
    n = np.sqrt(mu_earth / a**3)     # orbital angular rate
    u = n*t                          # argument of latitude

    # Position in ECI
    r_eci = a * np.array([
        np.cos(u),
        np.sin(u)*np.cos(inc),
        np.sin(u)*np.sin(inc)
    ])

    # Earth rotation to ECI to ECEF
    theta = wE * t
    Rz = np.array([
        [ np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [ 0,              0,             1]
    ])

    return Rz @ r_eci

def dipole_field_ecef(r_ecef):
    """ Dipole magnetic field in ECEF frame (Tesla). """
    r = np.linalg.norm(r_ecef)

    # Dipole tilt of 11 degrees
    tilt = np.deg2rad(11)
    Ry = np.array([
        [ np.cos(tilt), 0, np.sin(tilt)],
        [ 0,            1, 0],
        [-np.sin(tilt), 0, np.cos(tilt)]
    ])

    M_vec = Ry @ np.array([0, 0, M_EARTH])

    term1 = 3 * r_ecef * np.dot(M_vec, r_ecef) / r**5
    term2 = M_vec / r**3

    return 1e-7 * (term1 - term2)   # Convert to Tesla

def geomagnetic_field_body(q, t):
    """ Realistic B-field in body frame. """
    r_ecef = orbit_position_ecef(t)
    B_ecef = dipole_field_ecef(r_ecef)
    R_bi = quat_to_dcm(q)          # inertial to body frame
    return R_bi @ B_ecef