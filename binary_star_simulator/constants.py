"""
Physical constants for binary star simulations.
All values in SI units unless otherwise specified.
"""

import numpy as np

# Fundamental Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8  # Speed of light (m/s)
AU = 1.496e11  # Astronomical Unit (m)
YEAR_SECONDS = 365.25 * 24 * 3600  # Seconds in a year
PARSEC = 3.086e16  # Parsec in meters

# Solar quantities (reference values)
M_SUN = 1.989e30  # Solar mass (kg)
R_SUN = 6.96e8  # Solar radius (m)
L_SUN = 3.828e26  # Solar luminosity (W)

# Useful ratios
AU_PER_PARSEC = AU / PARSEC
YEAR_PER_SECOND = 1.0 / YEAR_SECONDS

# Useful combinations for orbital mechanics
# Kepler's third law constant: P^2 = (4π^2 / GM) * a^3
KEPLER_CONSTANT = 4 * np.pi**2 / G

# For convenience in solar system units:
# If masses in solar masses, semi-major axis in AU, period in years:
# P^2 = a^3 / M (where M is total mass in solar masses)

def kepler_period(a, M_total):
    """
    Calculate orbital period using Kepler's third law.
    
    Parameters
    ----------
    a : float
        Semi-major axis (AU)
    M_total : float
        Total mass (solar masses)
    
    Returns
    -------
    float
        Period in years
    """
    return np.sqrt(a**3 / M_total)

def semi_major_axis(P, M_total):
    """
    Calculate semi-major axis from period.
    
    Parameters
    ----------
    P : float
        Period (years)
    M_total : float
        Total mass (solar masses)
    
    Returns
    -------
    float
        Semi-major axis in AU
    """
    return (P**2 * M_total)**(1/3)

def orbital_velocity(M_total, a, r):
    """
    Calculate orbital velocity at distance r from barycenter.
    
    Parameters
    ----------
    M_total : float
        Total mass (kg)
    a : float
        Semi-major axis (m)
    r : float
        Distance from focus (m)
    
    Returns
    -------
    float
        Orbital velocity (m/s)
    """
    return np.sqrt(G * M_total * (2/r - 1/a))

def escape_velocity(M, r):
    """
    Calculate escape velocity at distance r from mass M.
    
    Parameters
    ----------
    M : float
        Mass (kg)
    r : float
        Distance from center (m)
    
    Returns
    -------
    float
        Escape velocity (m/s)
    """
    return np.sqrt(2 * G * M / r)
