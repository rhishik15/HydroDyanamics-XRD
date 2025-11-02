# cooling.py - Radiative cooling module for black hole accretion simulation
"""
Implements radiative cooling processes for hydrodynamic simulations:
- Bremsstrahlung (free-free) cooling
- Inverse Compton cooling from soft photon source
- Synchrotron cooling
- Compton heating/cooling

Based on standard astrophysical cooling functions adapted for
pseudo-Newtonian black hole accretion simulations.
"""

import numpy as np
from math import pi, exp, sqrt, log

# Physical constants (CGS units)
CONST_C = 2.99792458e10          # Speed of light (cm/s)
CONST_H = 6.6260755e-27          # Planck constant (erg s)
CONST_K = 1.380658e-16           # Boltzmann constant (erg/K)
CONST_ME = 9.1093897e-28         # Electron mass (g)
CONST_MP = 1.6726231e-24         # Proton mass (g)
CONST_SIGMA_T = 6.6524e-25       # Thomson cross section (cm^2)
CONST_SIGMA_SB = 5.67051e-5      # Stefan-Boltzmann (erg/cm^2/s/K^4)
CONST_E = 4.8032068e-10          # Electron charge (esu)
CONST_A_RAD = 7.5646e-15         # Radiation constant (erg/cm^3/K^4)

# Conversion factors
KEV_TO_ERG = 1.60218e-9          # keV to erg
ERG_TO_KEV = 1.0 / KEV_TO_ERG
MSUN = 1.989e33                  # Solar mass (g)
YEAR = 3.15576e7                 # Year in seconds


class CoolingModule:
    """
    Handles all radiative cooling/heating processes for the simulation
    """
    
    def __init__(self, params):
        """
        Initialize cooling module with simulation parameters
        
        Parameters:
        -----------
        params : dict
            Configuration dictionary with physical parameters
        """
        self.GAMMA = params.get("GAMMA", 5.0/3.0)
        self.BH_mass = params.get("BH_mass", 1.0)  # In code units
        self.BH_mass_cgs = self.BH_mass * MSUN      # In grams
        
        # Schwarzschild radius (cm)
        self.r_g = 2.0 * 6.674e-8 * self.BH_mass_cgs / (CONST_C**2)
        
        # Keplerian disk parameters
        self.disk_present = params.get("add_keplerian_disk", False)
        self.disk_mdot = params.get("disk_mdot", 1.0)  # In 10^17 g/s
        self.disk_r_in = params.get("disk_r_in", 3.0)   # In r_g
        self.disk_r_out = params.get("disk_r_out", 300.0)  # In r_g
        
        # Cooling switches
        self.use_bremsstrahlung = params.get("use_bremsstrahlung", True)
        self.use_compton = params.get("use_compton", True)
        self.use_synchrotron = params.get("use_synchrotron", False)
        
        # Cooling time floor (to avoid stiff equations)
        self.t_cool_min = params.get("t_cool_min", 1e-6)
        
        print("Cooling module initialized:")
        print(f"  Bremsstrahlung: {self.use_bremsstrahlung}")
        print(f"  Compton cooling: {self.use_compton}")
        print(f"  Synchrotron: {self.use_synchrotron}")
        print(f"  Keplerian disk: {self.disk_present}")
        if self.disk_present:
            print(f"    Disk mdot: {self.disk_mdot} × 10^17 g/s")
            print(f"    Disk range: {self.disk_r_in} - {self.disk_r_out} r_g")
    
    
    def compute_keplerian_disk_temperature(self, r_cyl):
        """
        Compute temperature of Keplerian disk at cylindrical radius r_cyl
        
        Standard Shakura-Sunyaev disk temperature profile:
        T(r) ∝ M^(-1/2) * mdot^(1/4) * r^(-3/4)
        
        Parameters:
        -----------
        r_cyl : float or array
            Cylindrical radius in units of r_g
        
        Returns:
        --------
        T_disk : float or array
            Disk temperature in Kelvin
        """
        if not self.disk_present:
            return 0.0
        
        # Convert to physical units
        r_cm = r_cyl * self.r_g
        
        # Shakura-Sunyaev temperature (simplified)
        # T ~ 5×10^7 K × (M/10Msun)^(-1/2) × (mdot)^(1/4) × (r/r_g)^(-3/4)
        
        # Normalization for standard disk
        T_norm = 5.0e7  # K
        
        # Scale with black hole mass
        mass_factor = (self.BH_mass / 10.0)**(-0.5)
        
        # Scale with accretion rate
        mdot_factor = self.disk_mdot**(0.25)
        
        # Radial dependence
        r_factor = r_cyl**(-0.75)
        
        # Inner boundary correction (goes to zero at ISCO)
        r_isco = self.disk_r_in
        boundary_factor = (1.0 - sqrt(r_isco / r_cyl))**(0.25)
        boundary_factor = np.maximum(boundary_factor, 0.0)
        
        T_disk = T_norm * mass_factor * mdot_factor * r_factor * boundary_factor
        
        return T_disk
    
    
    def compute_soft_photon_energy_density(self, R, Z):
        """
        Compute energy density of soft photons from Keplerian disk
        
        Parameters:
        -----------
        R, Z : array
            Cylindrical coordinates (in r_g units)
        
        Returns:
        --------
        u_rad : array
            Radiation energy density (erg/cm^3)
        T_rad : array
            Characteristic radiation temperature (K)
        """
        if not self.disk_present:
            return np.zeros_like(R), np.zeros_like(R)
        
        Nr, Nz = R.shape
        u_rad = np.zeros_like(R)
        T_rad = np.zeros_like(R)
        
        # For each point, compute diluted radiation from disk
        for i in range(Nr):
            for j in range(Nz):
                r_cyl = R[i, j]
                z_val = Z[i, j]
                
                # Skip if too close to disk plane
                if abs(z_val) < 0.1:
                    continue
                
                # Distance to nearest disk point
                if r_cyl < self.disk_r_in or r_cyl > self.disk_r_out:
                    continue
                
                # Distance from disk midplane
                d = abs(z_val * self.r_g)  # In cm
                
                # Disk temperature at this radius
                T_disk_local = self.compute_keplerian_disk_temperature(r_cyl)
                
                # Dilution factor (geometric)
                # Radiation dilutes as 1/d^2 from disk surface
                dilution = (self.r_g / max(d, 0.1 * self.r_g))**2
                dilution = min(dilution, 1.0)
                
                # Radiation energy density (diluted blackbody)
                u_rad[i, j] = CONST_A_RAD * T_disk_local**4 * dilution
                T_rad[i, j] = T_disk_local * dilution**0.25
        
        return u_rad, T_rad
    
    
    def bremsstrahlung_cooling(self, rho, T_electron, composition='H'):
        """
        Compute bremsstrahlung (free-free) cooling rate
        
        Lambda_ff ~ n_e * n_i * T^(1/2) * g_ff
        
        Parameters:
        -----------
        rho : array
            Mass density (code units)
        T_electron : array
            Electron temperature (Kelvin)
        composition : str
            'H' for hydrogen, 'He' for helium
        
        Returns:
        --------
        cooling_rate : array
            Energy loss rate per unit volume (erg/cm^3/s)
        """
        if not self.use_bremsstrahlung:
            return np.zeros_like(rho)
        
        # Convert density to CGS (assuming code units normalize to some rho_0)
        # For now, assume rho is in units where rho=1 corresponds to a typical value
        rho_cgs = rho * 1e-10  # Typical low-density plasma
        
        # Number densities
        mu_e = 1.18  # Mean molecular weight per electron (for ionized H)
        n_e = rho_cgs / (mu_e * CONST_MP)  # Electron number density (cm^-3)
        
        # Ion number density (assuming fully ionized hydrogen)
        n_i = n_e  # For H plasma
        
        # Gaunt factor (approximate, valid for most conditions)
        g_ff = np.maximum(1.0, np.sqrt(3.0 / pi) * np.log(
            2.0 * CONST_K * T_electron / (13.6 * 1.6e-12)
        ))
        
        # Bremsstrahlung cooling rate (Rybicki & Lightman formula)
        # Lambda_ff = 1.4e-27 * T^(1/2) * n_e * n_i * g_ff  [erg cm^-3 s^-1]
        
        cooling_rate = 1.4e-27 * np.sqrt(T_electron) * n_e * n_i * g_ff
        
        # Floor to avoid numerical issues
        cooling_rate = np.maximum(cooling_rate, 0.0)
        
        return cooling_rate
    
    
    def compton_cooling(self, rho, T_electron, u_rad, T_rad):
        """
        Compute Compton cooling/heating rate
        
        Inverse Compton: cool electrons scatter soft photons to higher energy
        Compton: hot photons transfer energy to cool electrons
        
        Parameters:
        -----------
        rho : array
            Mass density (code units)
        T_electron : array
            Electron temperature (K)
        u_rad : array
            Radiation energy density (erg/cm^3)
        T_rad : array
            Radiation temperature (K)
        
        Returns:
        --------
        cooling_rate : array
            Energy exchange rate (erg/cm^3/s)
            Positive = cooling, Negative = heating
        """
        if not self.use_compton:
            return np.zeros_like(rho)
        
        # Convert density to CGS
        rho_cgs = rho * 1e-10
        mu_e = 1.18
        n_e = rho_cgs / (mu_e * CONST_MP)
        
        # Compton cooling/heating rate
        # dE/dt = (4/3) * sigma_T * c * n_e * u_rad * (4kT_e - 4kT_rad) / (m_e c^2)
        # Simplified: dE/dt ~ sigma_T * c * n_e * u_rad * (T_e - T_rad) / T_e
        
        # Compton y-parameter
        theta_e = CONST_K * T_electron / (CONST_ME * CONST_C**2)
        
        # Cooling rate (exact formula for non-relativistic case)
        cooling_rate = (4.0 * CONST_SIGMA_T * CONST_C * n_e * u_rad * 
                       (T_electron - T_rad) / T_electron)
        
        # For relativistic electrons (theta_e > 0.1), include correction
        mask_hot = theta_e > 0.1
        if np.any(mask_hot):
            # Relativistic correction factor
            rel_correction = (1.0 + 4.0 * theta_e[mask_hot])
            cooling_rate[mask_hot] *= rel_correction
        
        return cooling_rate
    
    
    def synchrotron_cooling(self, rho, T_electron, B_field):
        """
        Compute synchrotron cooling rate
        
        Only relevant if magnetic fields are present
        
        Parameters:
        -----------
        rho : array
            Mass density
        T_electron : array
            Electron temperature (K)
        B_field : array
            Magnetic field strength (Gauss)
        
        Returns:
        --------
        cooling_rate : array
            Synchrotron cooling rate (erg/cm^3/s)
        """
        if not self.use_synchrotron:
            return np.zeros_like(rho)
        
        # Convert density
        rho_cgs = rho * 1e-10
        mu_e = 1.18
        n_e = rho_cgs / (mu_e * CONST_MP)
        
        # Average electron energy
        E_e = 1.5 * CONST_K * T_electron  # Non-relativistic
        
        # For relativistic electrons
        theta_e = CONST_K * T_electron / (CONST_ME * CONST_C**2)
        mask_rel = theta_e > 0.1
        if np.any(mask_rel):
            gamma_e = 1.0 + theta_e[mask_rel]
            E_e[mask_rel] = gamma_e * CONST_ME * CONST_C**2
        
        # Synchrotron cooling rate per electron
        # P_syn = (4/3) * sigma_T * c * (v/c)^2 * gamma^2 * U_B
        U_B = B_field**2 / (8.0 * pi)  # Magnetic energy density
        
        beta = np.sqrt(1.0 - 1.0/(1.0 + theta_e)**2)  # v/c
        gamma_lorentz = 1.0 + theta_e
        
        P_syn = (4.0/3.0) * CONST_SIGMA_T * CONST_C * beta**2 * \
                gamma_lorentz**2 * U_B
        
        # Total cooling rate
        cooling_rate = n_e * P_syn
        
        return cooling_rate
    
    
    def compute_total_cooling(self, rho, p, R, Z, GAMMA=None, B_field=None):
        """
        Compute total cooling rate from all processes
        
        Parameters:
        -----------
        rho : array (Nr, Nz)
            Mass density (code units)
        p : array (Nr, Nz)
            Pressure (code units)
        R, Z : array (Nr, Nz)
            Cylindrical coordinates (in r_g)
        GAMMA : float
            Adiabatic index (default from config)
        B_field : array or None
            Magnetic field strength (Gauss)
        
        Returns:
        --------
        cooling_rate : array
            Total cooling rate (code units per timestep)
        diagnostics : dict
            Breakdown of cooling components
        """
        if GAMMA is None:
            GAMMA = self.GAMMA
        
        # Convert pressure to temperature
        # p = (rho * k * T) / (mu * m_p) for ideal gas
        # T = p * mu * m_p / (rho * k)
        # For code units, assume p and rho are related by temperature
        
        # Temperature in code units (assuming p ~ rho * T)
        T_code = p / (rho + 1e-20)
        
        # Convert to physical temperature (K)
        # Assuming code temperature unit is ~kT/(m_e c^2)
        T_phys_factor = CONST_ME * CONST_C**2 / CONST_K  # ~6e9 K
        T_electron = T_code * T_phys_factor
        T_electron = np.maximum(T_electron, 1e4)  # Floor at 10^4 K
        
        # Get soft photon field from disk
        u_rad, T_rad = self.compute_soft_photon_energy_density(R, Z)
        
        # Compute individual cooling rates
        Lambda_brems = self.bremsstrahlung_cooling(rho, T_electron)
        Lambda_compton = self.compton_cooling(rho, T_electron, u_rad, T_rad)
        
        if B_field is not None and self.use_synchrotron:
            Lambda_sync = self.synchrotron_cooling(rho, T_electron, B_field)
        else:
            Lambda_sync = np.zeros_like(rho)
        
        # Total cooling (erg/cm^3/s)
        Lambda_total_cgs = Lambda_brems + Lambda_compton + Lambda_sync
        
        # Convert to code units
        # Cooling rate affects energy: dE/dt = -Lambda * Volume
        # In code units, we need cooling per unit mass per unit time
        
        # Estimate code unit conversion
        # Energy density in code units ~ rho * v^2 ~ rho * c^2
        # Time unit ~ r_g / c
        
        t_code = self.r_g / CONST_C  # Code time unit (s)
        rho_code = 1e-10  # Reference density (g/cm^3)
        E_code = rho_code * CONST_C**2  # Energy density unit
        
        # Cooling rate in code units (per unit volume per unit time)
        cooling_rate_code = Lambda_total_cgs * t_code / E_code
        
        # Apply cooling time floor to avoid timestep issues
        # tau_cool = E / (dE/dt) must be > t_cool_min
        
        internal_energy = p / (GAMMA - 1.0)
        tau_cool = internal_energy / (cooling_rate_code + 1e-20)
        
        # Limit cooling rate where cooling time is too short
        mask_fast = tau_cool < self.t_cool_min
        if np.any(mask_fast):
            cooling_rate_code[mask_fast] = internal_energy[mask_fast] / self.t_cool_min
        
        # Diagnostics
        diagnostics = {
            'bremsstrahlung': Lambda_brems,
            'compton': Lambda_compton,
            'synchrotron': Lambda_sync,
            'total_cgs': Lambda_total_cgs,
            'total_code': cooling_rate_code,
            'cooling_time': tau_cool,
            'T_electron': T_electron,
            'T_radiation': T_rad
        }
        
        return cooling_rate_code, diagnostics
    
    
    def apply_cooling_to_pressure(self, p, rho, cooling_rate, dt, GAMMA=None):
        """
        Apply cooling to pressure field
        
        Parameters:
        -----------
        p : array
            Pressure (code units)
        rho : array
            Density (code units)
        cooling_rate : array
            Cooling rate (code units per time)
        dt : float
            Timestep (code units)
        GAMMA : float
            Adiabatic index
        
        Returns:
        --------
        p_new : array
            Updated pressure after cooling
        """
        if GAMMA is None:
            GAMMA = self.GAMMA
        
        # Energy loss
        dE = cooling_rate * dt
        
        # Update pressure (E = p/(gamma-1))
        p_new = p - (GAMMA - 1.0) * dE
        
        # Ensure positive pressure
        p_floor = 1e-10
        p_new = np.maximum(p_new, p_floor)
        
        return p_new
    
    
    def get_cooling_timescale(self, rho, p, R, Z):
        """
        Estimate cooling timescale at each point
        
        Parameters:
        -----------
        rho, p : array
            Density and pressure
        R, Z : array
            Coordinates
        
        Returns:
        --------
        t_cool : array
            Cooling timescale (code units)
        """
        cooling_rate, _ = self.compute_total_cooling(rho, p, R, Z)
        
        internal_energy = p / (self.GAMMA - 1.0)
        t_cool = internal_energy / (cooling_rate + 1e-20)
        
        return t_cool


def create_cooling_diagnostics(cooling_module, rho, p, R, Z, step=0):
    """
    Create diagnostic plots and statistics for cooling
    
    Parameters:
    -----------
    cooling_module : CoolingModule
        Cooling module instance
    rho, p : array
        Hydrodynamic variables
    R, Z : array
        Coordinates
    step : int
        Timestep number
    """
    _, diagnostics = cooling_module.compute_total_cooling(rho, p, R, Z)
    
    print(f"\n--- Cooling Diagnostics (step {step}) ---")
    print(f"  Electron temperature: [{np.min(diagnostics['T_electron']):.2e}, "
          f"{np.max(diagnostics['T_electron']):.2e}] K")
    print(f"  Radiation temperature: [{np.min(diagnostics['T_radiation']):.2e}, "
          f"{np.max(diagnostics['T_radiation']):.2e}] K")
    print(f"  Cooling time: [{np.min(diagnostics['cooling_time']):.2e}, "
          f"{np.max(diagnostics['cooling_time']):.2e}] code units")
    
    # Identify dominant cooling mechanism
    brems_tot = np.sum(np.abs(diagnostics['bremsstrahlung']))
    compton_tot = np.sum(np.abs(diagnostics['compton']))
    sync_tot = np.sum(np.abs(diagnostics['synchrotron']))
    
    total = brems_tot + compton_tot + sync_tot
    if total > 0:
        print(f"  Cooling budget:")
        print(f"    Bremsstrahlung: {100*brems_tot/total:.1f}%")
        print(f"    Compton: {100*compton_tot/total:.1f}%")
        print(f"    Synchrotron: {100*sync_tot/total:.1f}%")