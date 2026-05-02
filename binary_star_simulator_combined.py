"""
Binary Star Simulator - Combined Module
All components in a single executable file.
"""

import numpy as np
from scipy.optimize import fsolve

# ============================================================================
# CONSTANTS
# ============================================================================

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
KEPLER_CONSTANT = 4 * np.pi**2 / G


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


# ============================================================================
# ORBITAL ELEMENTS
# ============================================================================

class OrbitalElements:
    """
    Represents and converts between orbital element representations.
    """
    
    def __init__(self, a, e, i, Omega, w, M, M_total=1.0, GM=None):
        """
        Initialize orbital elements.
        
        Parameters
        ----------
        a : float
            Semi-major axis (AU or meters)
        e : float
            Eccentricity (0 <= e < 1)
        i : float
            Inclination (degrees)
        Omega : float
            Longitude of ascending node (degrees)
        w : float
            Argument of periapsis (degrees)
        M : float
            Mean anomaly (degrees)
        M_total : float
            Total mass (solar masses, default 1.0)
        GM : float, optional
            Gravitational parameter (default: G * M_total in SI units)
        """
        # Validate eccentricity
        if not (0 <= e < 1):
            raise ValueError(f"Eccentricity must be in [0, 1), got {e}")
        
        self.a = a  # Semi-major axis
        self.e = e  # Eccentricity
        self.i = np.radians(i)  # Inclination (radians)
        self.Omega = np.radians(Omega)  # Longitude of ascending node (radians)
        self.w = np.radians(w)  # Argument of periapsis (radians)
        self.M = np.radians(M)  # Mean anomaly (radians)
        self.M_total = M_total
        
        if GM is None:
            self.GM = G * M_total
        else:
            self.GM = GM
    
    def mean_anomaly_to_eccentric_anomaly(self, M=None, tol=1e-10):
        """
        Solve Kepler's equation: M = E - e*sin(E)
        
        Parameters
        ----------
        M : float, optional
            Mean anomaly (radians). If None, use self.M
        tol : float
            Convergence tolerance
        
        Returns
        -------
        float
            Eccentric anomaly E (radians)
        """
        if M is None:
            M = self.M
        
        E = M + self.e * np.sin(M)
        
        for _ in range(100):
            sin_E = np.sin(E)
            cos_E = np.cos(E)
            f = E - self.e * sin_E - M
            f_prime = 1 - self.e * cos_E
            
            if abs(f_prime) < 1e-15:
                break
            
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tol:
                return E_new
            
            E = E_new
        
        return E
    
    def eccentric_anomaly_to_true_anomaly(self, E=None):
        """
        Convert eccentric anomaly to true anomaly.
        
        Uses: tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        
        Parameters
        ----------
        E : float, optional
            Eccentric anomaly (radians)
        
        Returns
        -------
        float
            True anomaly nu (radians)
        """
        if E is None:
            E = self.mean_anomaly_to_eccentric_anomaly()
        
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E/2),
            np.sqrt(1 - self.e) * np.cos(E/2)
        )
        
        return nu
    
    def true_anomaly_to_eccentric_anomaly(self, nu):
        """
        Convert true anomaly to eccentric anomaly.
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        float
            Eccentric anomaly E (radians)
        """
        E = 2 * np.arctan2(
            np.sqrt(1 - self.e) * np.sin(nu/2),
            np.sqrt(1 + self.e) * np.cos(nu/2)
        )
        return E
    
    def get_true_anomaly(self):
        """Get current true anomaly."""
        return self.eccentric_anomaly_to_true_anomaly()
    
    def distance_to_primary(self, nu=None):
        """
        Calculate distance from primary focus (r = a(1-e^2) / (1 + e*cos(nu))).
        
        Parameters
        ----------
        nu : float, optional
            True anomaly (radians). If None, compute from current M
        
        Returns
        -------
        float
            Distance from primary focus
        """
        if nu is None:
            nu = self.get_true_anomaly()
        
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(nu))
        return r
    
    def to_cartesian(self):
        """
        Convert orbital elements to Cartesian coordinates.
        
        Returns
        -------
        tuple
            (x, y, z, vx, vy, vz) in orbital plane frame
        """
        nu = self.get_true_anomaly()
        
        # Position in orbital plane
        r = self.distance_to_primary(nu)
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        # Velocity in orbital plane
        n = np.sqrt(self.GM / self.a**3)
        r_dot = self.a * self.e * np.sin(nu) * n / np.sqrt(1 - self.e**2)
        nu_dot = n * (1 + self.e * np.cos(nu))**2 / (1 - self.e**2)**(3/2)
        
        vx_orb = r_dot * np.cos(nu) - r * nu_dot * np.sin(nu)
        vy_orb = r_dot * np.sin(nu) + r * nu_dot * np.cos(nu)
        vz_orb = 0
        
        # Apply rotation matrices for orbital inclination and orientation
        # 1. Rotate around z by argument of periapsis (w)
        cos_w = np.cos(self.w)
        sin_w = np.sin(self.w)
        x1 = x_orb * cos_w - y_orb * sin_w
        y1 = x_orb * sin_w + y_orb * cos_w
        z1 = z_orb
        
        vx1 = vx_orb * cos_w - vy_orb * sin_w
        vy1 = vx_orb * sin_w + vy_orb * cos_w
        vz1 = vz_orb
        
        # 2. Rotate around x by inclination (i)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        x2 = x1
        y2 = y1 * cos_i - z1 * sin_i
        z2 = y1 * sin_i + z1 * cos_i
        
        vx2 = vx1
        vy2 = vy1 * cos_i - vz1 * sin_i
        vz2 = vy1 * sin_i + vz1 * cos_i
        
        # 3. Rotate around z by longitude of ascending node (Omega)
        cos_Omega = np.cos(self.Omega)
        sin_Omega = np.sin(self.Omega)
        x3 = x2 * cos_Omega - y2 * sin_Omega
        y3 = x2 * sin_Omega + y2 * cos_Omega
        z3 = z2
        
        vx3 = vx2 * cos_Omega - vy2 * sin_Omega
        vy3 = vx2 * sin_Omega + vy2 * cos_Omega
        vz3 = vz2
        
        return np.array([x3, y3, z3, vx3, vy3, vz3])
    
    @staticmethod
    def from_cartesian(x, y, z, vx, vy, vz, GM):
        """
        Convert Cartesian coordinates to orbital elements.
        
        Parameters
        ----------
        x, y, z : float
            Position (meters)
        vx, vy, vz : float
            Velocity (m/s)
        GM : float
            Gravitational parameter
        
        Returns
        -------
        OrbitalElements
            Orbital elements object
        """
        r = np.array([x, y, z])
        v = np.array([vx, vy, vz])
        
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        eps = v_mag**2 / 2 - GM / r_mag
        a = -GM / (2 * eps)
        
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        e_vec = np.cross(v, h) / GM - r / r_mag
        e = np.linalg.norm(e_vec)
        
        i = np.arccos(h[2] / h_mag)
        
        n_vec = np.array([-h[1], h[0], 0])
        n_mag = np.linalg.norm(n_vec)
        
        if n_mag > 1e-10:
            Omega = np.arctan2(n_vec[1], n_vec[0])
        else:
            Omega = 0
        
        if n_mag > 1e-10 and e > 1e-10:
            w = np.arctan2(
                np.dot(e_vec, np.array([0, 0, 1])),
                np.dot(e_vec, n_vec)
            )
        else:
            w = 0
        
        if e > 1e-10:
            nu = np.arctan2(
                np.dot(h, np.cross(r, v)) / (h_mag * np.linalg.norm(np.cross(r, v))),
                np.dot(r, e_vec) / (r_mag * e)
            )
        else:
            nu = np.arctan2(y, x)
        
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu/2),
            np.sqrt(1 + e) * np.cos(nu/2)
        )
        M = E - e * np.sin(E)
        
        return OrbitalElements(
            a=a,
            e=e,
            i=np.degrees(i),
            Omega=np.degrees(Omega) % 360,
            w=np.degrees(w) % 360,
            M=np.degrees(M) % 360,
            GM=GM
        )


# ============================================================================
# INTEGRATOR
# ============================================================================

class Integrator:
    """Base class for numerical integrators."""
    
    def __init__(self, masses, G_val=G):
        """
        Initialize integrator.
        
        Parameters
        ----------
        masses : array_like
            Masses of bodies (kg)
        G_val : float
            Gravitational constant
        """
        self.masses = np.array(masses)
        self.n_bodies = len(masses)
        self.G = G_val
        self.softening = 1e-6
    
    def acceleration(self, state):
        """
        Calculate accelerations from gravitational forces.
        
        Parameters
        ----------
        state : array
            Flattened state vector [x1, y1, z1, x2, y2, z2, ...]
        
        Returns
        -------
        array
            Acceleration vector [ax1, ay1, az1, ax2, ay2, az2, ...]
        """
        acc = np.zeros_like(state)
        positions = state.reshape(self.n_bodies, 3)
        accelerations = np.zeros((self.n_bodies, 3))
        
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue
                
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                r_softened = np.sqrt(r**2 + self.softening**2)
                
                acceleration = -self.G * self.masses[j] * r_vec / r_softened**3
                accelerations[i] += acceleration
        
        return accelerations.flatten()
    
    def energy(self, state, velocity):
        """
        Calculate total mechanical energy.
        
        Parameters
        ----------
        state : array
            Position state
        velocity : array
            Velocity state
        
        Returns
        -------
        dict
            Dictionary with 'kinetic', 'potential', 'total' energy
        """
        positions = state.reshape(self.n_bodies, 3)
        velocities = velocity.reshape(self.n_bodies, 3)
        
        KE = 0.5 * np.sum(self.masses[:, np.newaxis] * velocities**2)
        
        PE = 0
        for i in range(self.n_bodies):
            for j in range(i+1, self.n_bodies):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                PE -= self.G * self.masses[i] * self.masses[j] / r
        
        return {
            'kinetic': KE,
            'potential': PE,
            'total': KE + PE
        }


class LeapfrogIntegrator(Integrator):
    """
    Symplectic leapfrog integrator.
    
    Advantages:
    - Excellent energy conservation
    - Time-reversible
    - Second-order accurate
    """
    
    def step(self, state, velocity, dt):
        """
        Perform one leapfrog integration step.
        
        Parameters
        ----------
        state : array
            Position state
        velocity : array
            Velocity state
        dt : float
            Time step
        
        Returns
        -------
        tuple
            (new_state, new_velocity)
        """
        acc = self.acceleration(state)
        velocity_half = velocity + 0.5 * dt * acc
        
        state_new = state + dt * velocity_half
        
        acc_new = self.acceleration(state_new)
        
        velocity_new = velocity_half + 0.5 * dt * acc_new
        
        return state_new, velocity_new
    
    def integrate(self, state, velocity, t_initial, t_final, dt):
        """
        Integrate orbit from t_initial to t_final.
        
        Parameters
        ----------
        state : array
            Initial position
        velocity : array
            Initial velocity
        t_initial : float
            Initial time
        t_final : float
            Final time
        dt : float
            Time step
        
        Yields
        ------
        tuple
            (time, state, velocity, energy_dict)
        """
        t = t_initial
        
        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t
            
            state, velocity = self.step(state, velocity, dt)
            t += dt
            
            energy = self.energy(state, velocity)
            yield t, state, velocity, energy


class RK4Integrator(Integrator):
    """
    Runge-Kutta 4th order integrator.
    
    Advantages:
    - Higher accuracy (4th order)
    - Good for short integrations
    
    Disadvantage:
    - Energy not as well conserved as leapfrog
    """
    
    def _derivatives(self, state, velocity):
        """Calculate derivatives [velocity, acceleration]."""
        acc = self.acceleration(state)
        return np.concatenate([velocity, acc])
    
    def step(self, state, velocity, dt):
        """
        Perform one RK4 integration step.
        
        Parameters
        ----------
        state : array
            Position state
        velocity : array
            Velocity state
        dt : float
            Time step
        
        Returns
        -------
        tuple
            (new_state, new_velocity)
        """
        y = np.concatenate([state, velocity])
        
        dy1 = self._derivatives(state, velocity)
        
        dy2 = self._derivatives(
            state + 0.5 * dt * dy1[:len(state)],
            velocity + 0.5 * dt * dy1[len(state):]
        )
        
        dy3 = self._derivatives(
            state + 0.5 * dt * dy2[:len(state)],
            velocity + 0.5 * dt * dy2[len(state):]
        )
        
        dy4 = self._derivatives(
            state + dt * dy3[:len(state)],
            velocity + dt * dy3[len(state):]
        )
        
        dy = (dy1 + 2*dy2 + 2*dy3 + dy4) / 6
        y_new = y + dt * dy
        
        state_new = y_new[:len(state)]
        velocity_new = y_new[len(state):]
        
        return state_new, velocity_new
    
    def integrate(self, state, velocity, t_initial, t_final, dt):
        """
        Integrate orbit from t_initial to t_final.
        
        Yields
        ------
        tuple
            (time, state, velocity, energy_dict)
        """
        t = t_initial
        
        while t < t_final:
            if t + dt > t_final:
                dt = t_final - t
            
            state, velocity = self.step(state, velocity, dt)
            t += dt
            
            energy = self.energy(state, velocity)
            yield t, state, velocity, energy


class AdaptiveIntegrator(LeapfrogIntegrator):
    """
    Adaptive timestep integrator using error estimation.
    """
    
    def __init__(self, masses, G_val=G, tolerance=1e-8):
        """
        Initialize adaptive integrator.
        
        Parameters
        ----------
        masses : array_like
            Masses of bodies
        G_val : float
            Gravitational constant
        tolerance : float
            Integration error tolerance
        """
        super().__init__(masses, G_val)
        self.tolerance = tolerance
    
    def step_adaptive(self, state, velocity, dt_max):
        """
        Perform adaptive timestep integration step.
        
        Uses error estimation between full step and two half steps.
        
        Parameters
        ----------
        state : array
            Position state
        velocity : array
            Velocity state
        dt_max : float
            Maximum allowed timestep
        
        Returns
        -------
        tuple
            (new_state, new_velocity, dt_used, error_estimate, new_dt)
        """
        dt = dt_max
        
        while True:
            state_full, velocity_full = self.step(state, velocity, dt)
            
            state_half, velocity_half = self.step(state, velocity, dt/2)
            state_half, velocity_half = self.step(state_half, velocity_half, dt/2)
            
            error = np.linalg.norm(state_full - state_half)
            
            if error < self.tolerance:
                new_dt = dt * min(2.0, max(0.5, self.tolerance / error))
                return state_half, velocity_half, dt, error, new_dt
            else:
                dt *= 0.5
                if dt < 1e-12:
                    raise RuntimeError("Timestep too small, integration failed")


# ============================================================================
# EXAMPLE USAGE / TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Binary Star Simulator - Combined Module Demo")
    print("=" * 70)
    
    # Test orbital elements calculations
    print("\n1. ORBITAL ELEMENTS TEST")
    print("-" * 70)
    
    # Create orbital elements for a solar-like binary
    orbit = OrbitalElements(
        a=1.0,           # 1 AU semi-major axis
        e=0.1,           # Eccentricity
        i=0,             # No inclination
        Omega=0,         # No ascending node
        w=0,             # No argument of periapsis
        M=0,             # At periapsis
        M_total=2.0      # 2 solar masses total
    )
    
    print(f"Semi-major axis: {orbit.a} AU")
    print(f"Eccentricity: {orbit.e}")
    print(f"Period: {kepler_period(1.0, 2.0):.4f} years")
    print(f"Distance to primary: {orbit.distance_to_primary(orbit.get_true_anomaly()):.6f} AU")
    
    # Convert to Cartesian
    cartesian = orbit.to_cartesian()
    print(f"\nCartesian coordinates (AU, AU/year):")
    print(f"Position: ({cartesian[0]:.6f}, {cartesian[1]:.6f}, {cartesian[2]:.6f})")
    print(f"Velocity: ({cartesian[3]:.6f}, {cartesian[4]:.6f}, {cartesian[5]:.6f})")
    
    # Test integrator
    print("\n2. LEAPFROG INTEGRATOR TEST")
    print("-" * 70)
    
    # Two-body system: Earth-Sun system (scaled)
    masses = np.array([1.989e30, 5.972e24])  # Sun and Earth masses
    
    integrator = LeapfrogIntegrator(masses)
    
    # Initial conditions: Earth at 1 AU, circular orbit
    state = np.array([1.496e11, 0, 0])  # 1 AU from Sun
    velocity = np.array([0, 29780, 0])  # Orbital velocity ~29.8 km/s
    
    print("Initial state:")
    print(f"Position: {state} m")
    print(f"Velocity: {velocity} m/s")
    
    # Integrate for 1 second
    dt = 1000  # 1000 second timestep
    t_final = 86400  # 1 day in seconds
    
    results = list(integrator.integrate(state, velocity, 0, t_final, dt))
    
    if results:
        t_final_result, state_final, velocity_final, energy_final = results[-1]
        print(f"\nAfter {t_final_result/86400:.2f} days:")
        print(f"Position: {state_final} m")
        print(f"Velocity: {velocity_final} m/s")
        print(f"Total Energy: {energy_final['total']:.6e} J")
        print(f"Energy conservation check: Initial vs Final")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
