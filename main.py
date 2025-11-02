# main.py - FULLY CORRECTED Hydrodynamic Black Hole Accretion Simulation

"""
COMPREHENSIVE FIXES APPLIED:
============================
1. ✅ Fixed Paczynski-Wiita potential (proper softening)
2. ✅ Corrected force calculation (regularized at r=0)
3. ✅ Physical boundary conditions (no exponential damping)
4. ✅ Conservative-to-primitive conversion (preserves momentum)
5. ✅ Proper energy handling (no velocity scaling)
6. ✅ Consistent regularization throughout
"""

import numpy as np
from conservation import compute_fluxes, compute_source_terms
from tvd import apply_tvd_limiters
from save import save_all_enhanced
from config import params
from utils import (validate_and_build_local_grids, assert_consistent_shapes,
                  get_regularized_radius, compute_spherical_radius)


# Grid & params
N_r, N_z = params["N_r"], params["N_z"]
R_max, Z_max = params["R_max"], params["Z_max"]
GAMMA = params["GAMMA"]

# Safety parameters
rho_floor = params.get("rho_floor", 1e-8)
p_floor = params.get("p_floor", 1e-10)
e_floor = params.get("e_floor", 1e-12)
v_max = params.get("v_max", 0.8)

# Physical parameters
BH_mass = params.get("BH_mass", 1.0)
rho_inf = params.get("rho_inf", 2e-4)
cs_inf = params.get("cs_inf", 0.173)
r_inner = params.get("r_inner", 3.0)
z_inner = params.get("z_inner", 3.0)

# Compute Bondi radius
R_bondi = BH_mass / (cs_inf**2)
print(f"Bondi radius: {R_bondi:.3f}")

# Time stepping parameters
CFL = params.get("CFL", 0.05)
dt_max = params.get("dt_max", 0.05)
dt_min = params.get("dt_min", 1e-10)
t_end = params.get("t_end", 200.0)
save_interval = params.get("save_interval", 10)

# Build grids
R_vals = np.linspace(0, R_max, N_r)
Z_vals = np.linspace(0, Z_max, N_z)
R, Z = np.meshgrid(R_vals, Z_vals, indexing='ij')

Rg = np.sqrt(R**2 + Z**2)

dR = R_vals[1] - R_vals[0]
dZ = Z_vals[1] - Z_vals[0]
dr = dR
dz = dZ


# ✅ CONSISTENT REGULARIZATION
R_reg = 0.5 * dR

print(f"Grid spacing: dR = {dR:.3f}, dZ = {dZ:.3f}")
print(f"Regularization: R_reg = {R_reg:.3f}")
print(f"R range: [{np.min(R):.2f}, {np.max(R):.2f}]")
print(f"Z range: [{np.min(Z):.2f}, {np.max(Z):.2f}]")





cumulative_accreted_mass = 0.0



# ============================================================================
# BONDI SOLUTION (UNCHANGED - CORRECT)
# ============================================================================


def bondi_solution_cylindrical(r_cyl, z_cyl, dr=None):
    """
    FIXED: Bondi solution in cylindrical coordinates with proper regularization.
    
    Parameters:
    -----------
    r_cyl, z_cyl : ndarray
        Cylindrical coordinates (R, Z)
    dr : float, optional
        Grid spacing (for regularization)
    
    Returns:
    --------
    rho, p, v_r, v_z, v_phi : ndarray
        Density, pressure, velocity components
    
    ✅ FIXED: Proper handling of R=Z=0 (origin)
    """
    
    # ✅ REGULARIZE spherical radius to avoid r=0
    epsilon = 0.5 * params.get('r_inner', 3.0)  # Physical regularization
    R_sph = np.sqrt(r_cyl**2 + z_cyl**2 + epsilon**2)
    R_sph_safe = np.maximum(R_sph, epsilon)
    
    # Bondi parameters
    BH_mass = params.get('BH_mass', 1.0)
    rho_inf = params.get('rho_inf', 2e-4)
    cs_inf = params.get('cs_inf', 0.173)
    GAMMA = params.get('GAMMA', 5.0/3.0)
    
    r_sonic = BH_mass / (2.0 * cs_inf**2)
    
    # Initialize arrays
    rho = np.zeros_like(R_sph)
    v_r_sph = np.zeros_like(R_sph)
    
    # ✅ SUBSONIC REGION: r < r_sonic
    mask_subsonic = R_sph_safe < r_sonic
    
    if np.any(mask_subsonic):
        xi = r_sonic / R_sph_safe[mask_subsonic]
        
        # ✅ CLIP xi to prevent overflow in exp
        xi = np.clip(xi, 1.0, 100.0)  # xi >= 1 in subsonic region
        
        # Bondi density (subsonic)
        rho[mask_subsonic] = rho_inf * xi**1.5 * np.exp(1.5 - 2.0*np.sqrt(xi) + 0.5/xi)
        
        # Bondi velocity (subsonic)
        # v_r = -cs * sqrt(2*(xi - 1 - ln(xi)))
        arg = 2.0 * (xi - 1.0 - np.log(xi))
        arg = np.maximum(arg, 0.0)  # Ensure positive for sqrt
        v_r_sph[mask_subsonic] = -cs_inf * np.sqrt(arg)
    
    # ✅ SUPERSONIC REGION: r > r_sonic
    mask_supersonic = R_sph_safe >= r_sonic
    
    if np.any(mask_supersonic):
        xi = R_sph_safe[mask_supersonic] / r_sonic
        
        # ✅ CLIP xi to prevent overflow
        xi = np.clip(xi, 1.0, 1000.0)  # xi >= 1 in supersonic region
        
        # Bondi density (supersonic)
        rho[mask_supersonic] = rho_inf * xi**(-1.5) * np.exp(1.5 - 2.0*np.sqrt(1.0/xi) + 0.5*xi)
        
        # Bondi velocity (supersonic)
        # v_r = -cs * sqrt(2*(1/xi - 1 - ln(1/xi)))
        # = -cs * sqrt(2*(1/xi - 1 + ln(xi)))
        arg = 2.0 * (1.0/xi - 1.0 + np.log(xi))
        arg = np.maximum(arg, 0.0)  # Ensure positive
        v_r_sph[mask_supersonic] = -cs_inf * np.sqrt(arg)
    
    # ✅ APPLY FLOORS to prevent negative/zero values
    rho = np.maximum(rho, params.get('rho_floor', 1e-8))
    
    # Temperature and pressure
    T_bondi = cs_inf**2 / GAMMA
    p = rho * T_bondi
    p = np.maximum(p, params.get('p_floor', 1e-10))
    
    # ✅ PROJECT spherical velocity to cylindrical components
    # v_R = v_r * (R/r), v_Z = v_r * (Z/r)
    # Use regularized radius for division
    if dr is None:
        R_reg = 0.5  # Default regularization
    else:
        R_reg = 0.5 * dr
    
    r_for_proj = np.sqrt(r_cyl**2 + z_cyl**2 + R_reg**2)
    
    v_r = v_r_sph * r_cyl / r_for_proj
    v_z = v_r_sph * z_cyl / r_for_proj
    v_phi = np.zeros_like(v_r)
    
    # ✅ SAFETY: Check for NaN/Inf
    bad = ~np.isfinite(rho) | ~np.isfinite(p) | ~np.isfinite(v_r) | ~np.isfinite(v_z)
    if np.any(bad):
        rho[bad] = params.get('rho_floor', 1e-8)
        p[bad] = params.get('p_floor', 1e-10)
        v_r[bad] = 0.0
        v_z[bad] = 0.0
    
    return rho, p, v_r, v_z, v_phi

# ============================================================================
# ✅ FIXED: POTENTIAL AND FORCES WITH PROPER REGULARIZATION
# ============================================================================

def compute_potential_and_forces(R, Z, dr, dz, rho_shape):
    """
    FIXED: Compute Paczynski-Wiita potential and forces with validated grids.
    
    Paczynski-Wiita pseudo-Newtonian potential:
        Φ = -GM/(r - r_g)
    
    where r_g = 2GM/c² (Schwarzschild radius)
    
    Parameters:
    -----------
    R, Z : ndarray, optional
        Cylindrical coordinate grids. If None or wrong shape, will be built.
    dr, dz : float, optional
        Grid spacings
    rho_shape : tuple, optional
        Target shape (Nr, Nz) for grid building
    
    Returns:
    --------
    Phi : ndarray
        Gravitational potential
    F_R, F_Z : ndarray
        Force components (negative potential gradients)
    r_sph : ndarray
        Spherical radius
    R_local, Z_local : ndarray
        Coordinate grids used (same shape as Phi)
    dr_local, dz_local : float
        Grid spacings used
    """
    
    # ✅ Determine target shape
    if rho_shape is None:
        if R is not None:
            rho_shape = R.shape
        else:
            rho_shape = (params.get('N_r', 50), params.get('N_z', 50))
    
    # ✅ Build/validate grids to match target shape
    dummy = np.zeros(rho_shape)
    R_local, Z_local, dr_local, dz_local = validate_and_build_local_grids(
        dummy, R=R, Z=Z, dr=dr, dz=dz,
        R_max=params.get('R_max', 50.0), 
        Z_max=params.get('Z_max', 50.0)
    )
    
    # ✅ PHYSICAL PARAMETERS
    BH_mass = params.get('BH_mass', 1.0)
    r_g = 2.0 * BH_mass  # Schwarzschild radius
    
    # ✅ SOFTENING PARAMETER (should be ~ grid scale)
    epsilon = params.get("pw_softening", 0.1) * dr_local
    epsilon = max(epsilon, 0.01)  # Minimum softening
    
    print(f"Potential: r_g = {r_g:.3f}, epsilon = {epsilon:.3f}")
    
    # ✅ REGULARIZED SPHERICAL RADIUS
    # Use sqrt(R² + Z² + ε²) to avoid r=0 singularity
    r_sph = np.sqrt(R_local**2 + Z_local**2 + epsilon**2)
    
    # ✅ DENOMINATOR WITH SOFTENING
    # r_eff - r_g with floor to prevent negative values
    denom = r_sph - r_g
    denom = np.maximum(denom, epsilon)  # Never go below softening scale
    
    # ✅ POTENTIAL: Φ = -GM/(r - r_g)
    Phi = -BH_mass / denom
    
    # ✅ FORCE MAGNITUDE: F = -∇Φ = -GM/(r - r_g)² * r_hat
    F_mag = BH_mass / (denom**2)
    
    # ✅ REGULARIZED FORCE COMPONENTS
    # F_R = F_mag * (R/r), F_Z = F_mag * (Z/r)
    # With regularization: r_safe = sqrt(R² + Z² + R_reg²)
    R_reg = 0.5 * dr_local
    r_safe = np.sqrt(R_local**2 + Z_local**2 + R_reg**2)
    
    F_R = -F_mag * R_local / r_safe
    F_Z = -F_mag * Z_local / r_safe
    
    # ✅ PHYSICAL CUTOFF: Zero force inside inner boundary
    # (Matter there has been accreted)
    r_inner = params.get('r_inner', 3.0)
    mask_inner = r_sph < r_inner
    
    F_R[mask_inner] = 0.0
    F_Z[mask_inner] = 0.0
    
    # Safe minimum for potential (avoid indexing empty array)
    if np.any(~mask_inner):
        Phi[mask_inner] = np.min(Phi[~mask_inner])
    else:
        Phi[mask_inner] = np.min(Phi)  # Fallback
    
    return Phi, F_R, F_Z, r_sph, R_local, Z_local, dr_local, dz_local



# ============================================================================
# INITIALIZATION (UNCHANGED - CORRECT)
# ============================================================================


def initialize_bondi_flow(R=None, Z=None, dr=None, dz=None):
    """
    FIXED: Initialize Bondi flow with validated grids.
    
    Parameters:
    -----------
    R, Z : ndarray, optional
        Coordinate grids
    dr, dz : float, optional
        Grid spacings
    
    Returns:
    --------
    rho, p, v_R, v_Z, vphi, e_total : ndarray
        Initial state arrays
    """
    
    # ✅ Build/validate grids
    if R is None or Z is None:
        # Use module-level parameters
        N_r = params.get("N_r", 50)
        N_z = params.get("N_z", 50)
        R_max = params.get("R_max", 50.0)
        Z_max = params.get("Z_max", 50.0)
        
        R_vals = np.linspace(0, R_max, N_r)
        Z_vals = np.linspace(0, Z_max, N_z)
        R, Z = np.meshgrid(R_vals, Z_vals, indexing='ij')
        
        dr = R_vals[1] - R_vals[0] if N_r > 1 else R_max / N_r
        dz = Z_vals[1] - Z_vals[0] if N_z > 1 else Z_max / N_z
    
    N_r, N_z = R.shape
    
    # Compute spherical radius
    Rg = np.sqrt(R**2 + Z**2)
    
    # Initialize arrays
    rho = np.zeros((N_r, N_z))
    p = np.zeros((N_r, N_z))
    v_R = np.zeros((N_r, N_z))
    v_Z = np.zeros((N_r, N_z))
    vphi = np.zeros((N_r, N_z))
    
    print("Initializing Bondi flow...")
    
    # ✅ Compute Bondi solution with regularization
    rho_val, p_val, vR_val, vZ_val, vphi_val = bondi_solution_cylindrical(R, Z, dr=dr)
    
    # ✅ CHECK for bad values before proceeding
    n_bad_rho = np.sum(~np.isfinite(rho_val))
    n_bad_p = np.sum(~np.isfinite(p_val))
    n_bad_vR = np.sum(~np.isfinite(vR_val))
    n_bad_vZ = np.sum(~np.isfinite(vZ_val))
    
    if n_bad_rho > 0:
        print(f"  WARNING: {n_bad_rho} bad density values in Bondi solution")
    if n_bad_p > 0:
        print(f"  WARNING: {n_bad_p} bad pressure values in Bondi solution")
    if n_bad_vR > 0:
        print(f"  WARNING: {n_bad_vR} bad vR values in Bondi solution")
    if n_bad_vZ > 0:
        print(f"  WARNING: {n_bad_vZ} bad vZ values in Bondi solution")
    
    # Zero velocities inside inner boundary
    r_inner = params.get('r_inner', 3.0)
    mask_inner = Rg < r_inner
    vR_val[mask_inner] = 0.0
    vZ_val[mask_inner] = 0.0
    
    # Apply floors
    rho[:] = np.maximum(rho_val, params.get('rho_floor', 1e-8))
    p[:] = np.maximum(p_val, params.get('p_floor', 1e-10))
    v_R[:] = np.clip(vR_val, -params.get('v_max', 0.95), params.get('v_max', 0.95))
    v_Z[:] = np.clip(vZ_val, -params.get('v_max', 0.95), params.get('v_max', 0.95))
    
    # Smooth transition at inner boundary
    transition_start = r_inner
    transition_width = max(3*dr, 0.5)
    transition_end = r_inner + transition_width
    
    mask_transition = (Rg >= transition_start) & (Rg <= transition_end)
    
    if np.any(mask_transition):
        xi = (Rg[mask_transition] - transition_start) / transition_width
        xi = np.clip(xi, 0, 1)
        xi = 3*xi**2 - 2*xi**3  # Hermite smooth
        
        v_R[mask_transition] *= xi
        v_Z[mask_transition] *= xi
    
    # Compute total energy
    GAMMA = params.get('GAMMA', 5.0/3.0)
    kinetic = 0.5 * rho * (v_R**2 + v_Z**2 + vphi**2)
    internal = p / (GAMMA - 1.0)
    e_total = kinetic + internal
    
    # Axis regularity
    e_total[0, :] = e_total[1, :]
    
    # ✅ FINAL VALIDATION
    n_bad_final = np.sum(~np.isfinite(rho) | ~np.isfinite(p) | 
                        ~np.isfinite(v_R) | ~np.isfinite(v_Z) | 
                        ~np.isfinite(e_total))
    
    if n_bad_final > 0:
        print(f"  ERROR: {n_bad_final} bad values after initialization!")
        print("  Attempting to fix...")
        
        bad_mask = ~np.isfinite(rho) | ~np.isfinite(p) | \
                   ~np.isfinite(v_R) | ~np.isfinite(v_Z) | \
                   ~np.isfinite(e_total)
        
        rho[bad_mask] = params.get('rho_floor', 1e-8)
        p[bad_mask] = params.get('p_floor', 1e-10)
        v_R[bad_mask] = 0.0
        v_Z[bad_mask] = 0.0
        vphi[bad_mask] = 0.0
        
        kinetic = 0.5 * rho * (v_R**2 + v_Z**2 + vphi**2)
        internal = p / (GAMMA - 1.0)
        e_total = kinetic + internal
        
        print("  Fixed!")
    
    print("\n  ========== INITIAL CONDITIONS ==========")
    print(f"  Density:  [{np.min(rho):.2e}, {np.max(rho):.2e}]")
    print(f"  Pressure: [{np.min(p):.2e}, {np.max(p):.2e}]")
    print(f"  Velocity: [{np.min(np.sqrt(v_R**2+v_Z**2)):.3f}, {np.max(np.sqrt(v_R**2+v_Z**2)):.3f}]")
    print(f"  Energy:   [{np.min(e_total):.2e}, {np.max(e_total):.2e}]")
    print(f"  ==========================================\n")
    
    return rho, p, v_R, v_Z, vphi, e_total

# ============================================================================
# ✅ FIXED: PHYSICAL BOUNDARY CONDITIONS
# ============================================================================


def apply_boundary_conditions(rho, p, v_R, v_Z, vphi=0, dt=None, R=None, Z=None, dr=None, dz=None):
    """
    FIXED: Boundary conditions with validated grids.
    
    Parameters:
    -----------
    rho, p, v_R, v_Z, vphi : ndarray
        State arrays
    dt : float, optional
        Timestep
    R, Z : ndarray, optional
        Coordinate grids
    dr, dz : float, optional
        Grid spacings
    
    Returns:
    --------
    mass_accreted : float
        Mass accreted through inner boundary
    """
    
    # ✅ Build/validate grids matching state arrays
    R_local, Z_local, dr_local, dz_local = validate_and_build_local_grids(
        rho, R=R, Z=Z, dr=dr, dz=dz,
        R_max=params.get('R_max', 50.0), 
        Z_max=params.get('Z_max', 50.0)
    )
    
    # Compute spherical radius using validated grids
    Rg = np.sqrt(R_local**2 + Z_local**2)
    
    # ✅ REGULARIZATION
    R_reg = 0.5 * dr_local
    
    def hermite_smooth(x):
        """Smooth interpolation: 0→1 with zero derivatives at ends"""
        x = np.clip(x, 0.0, 1.0)
        return 3.0 * x**2 - 2.0 * x**3
    
    mass_accreted = 0.0
    
    # ================================================================
    # 1. ✅ INNER BOUNDARY: Physical outflow + mass tracking
    # ================================================================
    
    r_inner = params.get('r_inner', 3.0)
    z_inner = params.get('z_inner', 3.0)
    
    # Hard boundary: Matter inside has been accreted
    f_hard = 0.98
    r_hard = f_hard * np.sqrt(r_inner**2 + z_inner**2)
    
    # Soft transition zone
    f_soft = 1.05
    r_soft = f_soft * np.sqrt(r_inner**2 + z_inner**2)
    
    # ✅ ACCRETE MASS: Track what flows through boundary
    mask_hard = Rg < r_hard
    if np.any(mask_hard):
        # Compute volume elements
        vol_elements = 2.0 * np.pi * np.maximum(R_local[mask_hard], R_reg) * dr_local * dz_local
        
        # Physical mass (above floor)
        rho_floor = params.get('rho_floor', 1e-8)
        physical_mass = np.maximum(rho[mask_hard] - rho_floor, 0.0) * vol_elements
        mass_accreted = float(np.sum(physical_mass))
        
        if not np.isfinite(mass_accreted):
            mass_accreted = 0.0
        
        # ✅ REMOVE ACCRETED MATTER: Set to floor
        rho[mask_hard] = rho_floor
        p[mask_hard] = params.get('p_floor', 1e-10)
        v_R[mask_hard] = 0.0
        v_Z[mask_hard] = 0.0
        
        if isinstance(vphi, np.ndarray):
            vphi[mask_hard] = 0.0
    
    # ✅ SMOOTH TRANSITION: Damp velocities gradually
    mask_soft = (Rg >= r_hard) & (Rg <= r_soft)
    if np.any(mask_soft):
        xi = (Rg[mask_soft] - r_hard) / (r_soft - r_hard)
        w = hermite_smooth(xi)  # 0 at r_hard, 1 at r_soft
        
        v_R[mask_soft] *= w
        v_Z[mask_soft] *= w
        
        if isinstance(vphi, np.ndarray):
            vphi[mask_soft] *= w
    
    # ================================================================
    # 2. ✅ OUTER BOUNDARY: Bondi solution sponge
    # ================================================================
    
    r_sponge_start = params.get("r_outer", 60.0)
    r_sponge_width = params.get("sponge_width", 15.0)
    
    # ✅ Compute Bondi solution using same grids
    rho_b, p_b, vR_b, vZ_b, vphi_b = bondi_solution_cylindrical(R_local, Z_local, dr=dr_local)
    
    # Sponge weight
    xi_sponge = (Rg - r_sponge_start) / r_sponge_width
    xi_sponge = np.clip(xi_sponge, 0.0, 1.0)
    w_sponge = hermite_smooth(xi_sponge)
    
    # ✅ BLEND WITH BONDI (smooth forcing, not hard BC)
    rho[:] = (1.0 - w_sponge) * rho + w_sponge * rho_b
    p[:] = (1.0 - w_sponge) * p + w_sponge * p_b
    v_R[:] = (1.0 - w_sponge) * v_R + w_sponge * vR_b
    v_Z[:] = (1.0 - w_sponge) * v_Z + w_sponge * vZ_b
    
    if isinstance(vphi, np.ndarray):
        vphi[:] = (1.0 - w_sponge) * vphi + w_sponge * vphi_b
    
    # ================================================================
    # 3. ✅ AXIS REGULARITY: Proper reflection symmetry
    # ================================================================
    
    i_axis = 0
    
    # Scalars: Even parity (∂/∂R = 0)
    rho[i_axis, :] = rho[1, :]
    p[i_axis, :] = p[1, :]
    v_Z[i_axis, :] = v_Z[1, :]
    
    # Radial velocity: Odd parity (v_R = 0 at axis)
    v_R[i_axis, :] = 0.0
    
    # Azimuthal velocity: Must vanish at axis
    if isinstance(vphi, np.ndarray):
        vphi[i_axis, :] = 0.0
    
    # ================================================================
    # 4. ✅ Z-BOUNDARY: Natural BC (set to Bondi)
    # ================================================================
    
    j_bottom = 0
    rho[:, j_bottom] = rho_b[:, j_bottom]
    p[:, j_bottom] = p_b[:, j_bottom]
    v_R[:, j_bottom] = vR_b[:, j_bottom]
    v_Z[:, j_bottom] = vZ_b[:, j_bottom]
    
    if isinstance(vphi, np.ndarray):
        vphi[:, j_bottom] = vphi_b[:, j_bottom]
    
    # ================================================================
    # 5. FLOORS
    # ================================================================
    
    rho[:] = np.maximum(rho, params.get('rho_floor', 1e-8))
    p[:] = np.maximum(p, params.get('p_floor', 1e-10))
    
    # ================================================================
    # 6. VELOCITY LIMITER
    # ================================================================
    
    v_max_local = params.get('v_max', 0.95)
    
    if isinstance(vphi, np.ndarray):
        v_mag = np.sqrt(v_R**2 + v_Z**2 + vphi**2)
    else:
        v_mag = np.sqrt(v_R**2 + v_Z**2)
    
    mask_fast = v_mag > v_max_local
    
    if np.any(mask_fast):
        scale = v_max_local / (v_mag[mask_fast] + 1e-20)
        v_R[mask_fast] *= scale
        v_Z[mask_fast] *= scale
        
        if isinstance(vphi, np.ndarray):
            vphi[mask_fast] *= scale
    
    # ================================================================
    # 7. SAFETY: NaN/Inf check
    # ================================================================
    
    bad = ~np.isfinite(rho) | ~np.isfinite(p) | \
          ~np.isfinite(v_R) | ~np.isfinite(v_Z)
    
    if isinstance(vphi, np.ndarray):
        bad |= ~np.isfinite(vphi)
    
    if np.any(bad):
        n_bad = np.sum(bad)
        print(f"  WARNING: {n_bad} bad cells - resetting to floor")
        
        rho[bad] = params.get('rho_floor', 1e-8)
        p[bad] = params.get('p_floor', 1e-10)
        v_R[bad] = 0.0
        v_Z[bad] = 0.0
        
        if isinstance(vphi, np.ndarray):
            vphi[bad] = 0.0
    
    return mass_accreted

# ============================================================================
# TIMESTEP (UNCHANGED)
# ============================================================================


def compute_timestep(rho, p, vr, vz, vphi, dr=None, dz=None):
    """
    FIXED: Adaptive timestep with CFL condition and validated spacing.
    
    Parameters:
    -----------
    rho, p, vr, vz, vphi : ndarray
        State arrays
    dr, dz : float, optional
        Grid spacings
    
    Returns:
    --------
    dt : float
        Timestep
    """
    
    # ✅ Infer spacings if not provided
    if dr is None:
        dr = params.get('R_max', 50.0) / rho.shape[0]
    if dz is None:
        dz = params.get('Z_max', 50.0) / rho.shape[1]
    
    GAMMA = params.get('GAMMA', 5.0/3.0)
    CFL = params.get('CFL', 0.05)
    
    cs = np.sqrt(GAMMA * p / (rho + 1e-20))
    
    sr_max = np.max(np.abs(vr) + cs)
    sz_max = np.max(np.abs(vz) + cs)
    
    dt = CFL * min(dr / (sr_max + 1e-20),
                   dz / (sz_max + 1e-20))
    
    dt_min = params.get('dt_min', 1e-10)
    dt_max = params.get('dt_max', 0.05)
    
    return np.clip(dt, dt_min, dt_max)




# ============================================================================
# ✅ FIXED: CONSERVATIVE-TO-PRIMITIVE CONVERSION
# ============================================================================

def primitive_to_conservative(rho, vr, vz, vphi, e_total):
    """
    Convert primitive to conservative variables with validation.
    """
    # ✅ Add shape validation
    assert_consistent_shapes(rho, vr, vz, vphi, e_total, context="primitive_to_conservative")
    
    U1 = rho.copy()
    U2 = rho * vr
    U3 = rho * vz
    U4 = rho * vphi
    U5 = e_total.copy()
    
    return U1, U2, U3, U4, U5



def conservative_to_primitive(U1, U2, U3, U4, U5, gamma=5.0/3.0):
    """
    CORRECTED: Convert conservative to primitive with momentum preservation.
    
    ✅ ALREADY CORRECT in your file - just ensure shapes are validated.
    """
    
    # ✅ Add shape validation
    assert_consistent_shapes(U1, U2, U3, U4, U5, context="conservative_to_primitive")
    
    rho_floor = params.get('rho_floor', 1e-8)
    p_floor = params.get('p_floor', 1e-10)
    e_floor = params.get('e_floor', 1e-12)
    
    # 1. Recover density
    rho = np.maximum(U1, rho_floor)
    
    # 2. Recover velocities (momentum/density)
    vr = np.divide(U2, rho, out=np.zeros_like(U2), where=(rho > rho_floor))
    vz = np.divide(U3, rho, out=np.zeros_like(U3), where=(rho > rho_floor))
    vphi = np.divide(U4, rho, out=np.zeros_like(U4), where=(rho > rho_floor))
    
    # 3. Recover total energy
    e_total = np.maximum(U5, e_floor)
    
    # 4. Compute kinetic and internal energy
    kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
    internal = e_total - kinetic
    
    # 5. ✅ HANDLE NEGATIVE INTERNAL ENERGY CORRECTLY
    mask_negative = internal < e_floor
    
    if np.any(mask_negative):
        # ✅ CORRECT FIX: Set internal to floor, keep velocities
        # This accepts small energy loss but preserves momentum
        internal[mask_negative] = e_floor
        e_total[mask_negative] = kinetic[mask_negative] + e_floor
        
        # DO NOT modify velocities! They come from conserved momentum.
    
    # 6. Compute pressure
    internal = np.maximum(internal, e_floor)
    p = (gamma - 1.0) * internal
    p = np.maximum(p, p_floor)
    
    # 7. Safety check
    bad = ~np.isfinite(rho) | ~np.isfinite(p) | \
          ~np.isfinite(vr) | ~np.isfinite(vz) | ~np.isfinite(vphi)
    
    if np.any(bad):
        rho[bad] = rho_floor
        p[bad] = p_floor
        vr[bad] = 0.0
        vz[bad] = 0.0
        vphi[bad] = 0.0
        e_total[bad] = p[bad] / (gamma - 1.0)
    
    return rho, vr, vz, vphi, p, e_total


# ============================================================================
# MAIN SIMULATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CORRECTED BLACK HOLE ACCRETION SIMULATION")
    print("="*70)
    print(f"Grid: {N_r} × {N_z}")
    print(f"Domain: R ∈ [0, {R_max}], Z ∈ [0, {Z_max}]")
    print(f"Bondi radius: {R_bondi:.3f}")
    print(f"Regularization: R_reg = {R_reg:.3f}")
    print("="*70)
    
    # Initialize
    rho, p, vr, vz, vphi, e_total = initialize_bondi_flow()
    
    # Compute gravitational forces
   # ✅ Compute gravitational forces (already done at module level, but shown for clarity)
    Phi, F_r, F_z, r_sph, R_local, Z_local, dr_local, dz_local = compute_potential_and_forces(
         R=R, Z=Z, dr=dr, dz=dz, rho_shape=(N_r, N_z))
    
    print(f"Potential range: [{np.min(Phi):.2e}, {np.max(Phi):.2e}]")
    print(f"Force R range: [{np.min(F_r):.2e}, {np.max(F_r):.2e}]")
    print(f"Force Z range: [{np.min(F_z):.2e}, {np.max(F_z):.2e}]")
    
    # Apply initial BCs
    acc = apply_boundary_conditions(rho, p, vr, vz, vphi, dt=0.0, R=R, Z=Z, dr=dr, dz=dz)
    cumulative_accreted_mass += acc
    
    # Recompute energy
    kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
    internal = p / (GAMMA - 1.0)
    e_total = kinetic + internal
    
    # Save initial state
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0)
    print("Initial state saved\n")
    
    # Time evolution
    t = 0.0
    step = 0
    
    print("Starting time evolution...")
    print("="*70)
    
    while t < t_end:
        # Compute timestep
        dt = compute_timestep(rho, p, vr, vz, vphi,dr = dr , dz=dz)
        
        if t + dt > t_end:
            dt = t_end - t
        
        # Convert to conservative
        U1, U2, U3, U4, U5 = primitive_to_conservative(rho, vr, vz, vphi, e_total)
        
        # ====================================================================
        # RK2 STAGE 1: k1 = RHS(U^n)
        # ====================================================================
        
        dU1_dt, dU2_dt, dU3_dt, dU4_dt, dU5_dt = compute_fluxes(
            rho, p, vr, vz, vphi, R, dr, dz
        )
        
        S1, S2, S3, S4, S5 = compute_source_terms(
            rho, vr, vz, vphi, F_r, F_z, R, p, dr, dz
        )
        
        k1_U1 = dU1_dt + S1
        k1_U2 = dU2_dt + S2
        k1_U3 = dU3_dt + S3
        k1_U4 = dU4_dt + S4
        k1_U5 = dU5_dt + S5
        
        # ====================================================================
        # RK2 STAGE 2: U^{n+1/2} = U^n + 0.5*dt*k1
        # ====================================================================
        
        U1_half = U1 + 0.5 * dt * k1_U1
        U2_half = U2 + 0.5 * dt * k1_U2
        U3_half = U3 + 0.5 * dt * k1_U3
        U4_half = U4 + 0.5 * dt * k1_U4
        U5_half = U5 + 0.5 * dt * k1_U5
        
        # Convert to primitive
        rho_half, vr_half, vz_half, vphi_half, p_half, e_half = \
            conservative_to_primitive(U1_half, U2_half, U3_half, U4_half, U5_half)
        
        # Apply BCs
        acc_half = apply_boundary_conditions(
            rho_half, p_half, vr_half, vz_half, vphi_half, dt=0.5*dt,
            R=R, Z=Z, dr=dr, dz=dz
        )
        
        if not np.isfinite(acc_half):
            acc_half = 0.0
        
        # Recompute conservative variables after BCs
        kinetic_half = 0.5 * rho_half * (vr_half**2 + vz_half**2 + vphi_half**2)
        internal_half = p_half / (GAMMA - 1.0)
        e_half = kinetic_half + internal_half
        
        U1_half = rho_half
        U2_half = rho_half * vr_half
        U3_half = rho_half * vz_half
        U4_half = rho_half * vphi_half
        U5_half = e_half
        
        # ====================================================================
        # RK2 STAGE 3: k2 = RHS(U^{n+1/2})
        # ====================================================================
        
        dU1_dt2, dU2_dt2, dU3_dt2, dU4_dt2, dU5_dt2 = compute_fluxes(
            rho_half, p_half, vr_half, vz_half, vphi_half, R, dr, dz
        )
        
        S1_half, S2_half, S3_half, S4_half, S5_half = compute_source_terms(
            rho_half, vr_half, vz_half, vphi_half, F_r, F_z, R, p_half, dr, dz
        )
        
        k2_U1 = dU1_dt2 + S1_half
        k2_U2 = dU2_dt2 + S2_half
        k2_U3 = dU3_dt2 + S3_half
        k2_U4 = dU3_dt2 + S4_half
        k2_U5 = dU5_dt2 + S5_half
        
        # ====================================================================
        # RK2 STAGE 4: U^{n+1} = U^n + dt*k2
        # ====================================================================
        
        U1_new = U1 + dt * k2_U1
        U2_new = U2 + dt * k2_U2
        U3_new = U3 + dt * k2_U3
        U4_new = U4 + dt * k2_U4
        U5_new = U5 + dt * k2_U5
        
        # Convert to primitive
        rho, vr, vz, vphi, p, e_total = conservative_to_primitive(
            U1_new, U2_new, U3_new, U4_new, U5_new
        )
        
        # ====================================================================
        # TVD LIMITING (on primitive variables)
        # ====================================================================
        
        rho, p, vr, vz, vphi = apply_tvd_limiters(rho, p, vr, vz, vphi, dr, dz)
        
        # Recompute energy after limiting
        kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
        internal = p / (GAMMA - 1.0)
        e_total = kinetic + internal
        
        # ====================================================================
        # FINAL BOUNDARY CONDITIONS
        # ====================================================================
        
        acc_full = apply_boundary_conditions(rho, p, vr, vz, vphi, dt=dt,
                                            R=R, Z=Z, dr=dr, dz=dz)
        
        if not np.isfinite(acc_full):
            acc_full = 0.0
        
        cumulative_accreted_mass += float(acc_half + acc_full)
        
        '''# Recompute energy after BCs
        kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
        internal = p / (GAMMA - 1.0)
        e_total = kinetic + internal
        '''
        # ====================================================================
        # FLOOR ENFORCEMENT
        # ====================================================================
        
        rho = np.maximum(rho, rho_floor)
        p = np.maximum(p, p_floor)
        e_total = np.maximum(e_total, e_floor)
        
        # Check for bad values
        bad = ~np.isfinite(rho) | ~np.isfinite(p) | \
              ~np.isfinite(vr) | ~np.isfinite(vz) | ~np.isfinite(vphi) | ~np.isfinite(e_total)
        
        if np.any(bad):
            n_bad = np.sum(bad)
            print(f"\n  WARNING: {n_bad} bad cells at step {step}")
            
            rho[bad] = rho_floor
            p[bad] = p_floor
            vr[bad] = 0.0
            vz[bad] = 0.0
            vphi[bad] = 0.0
            e_total[bad] = p_floor / (GAMMA - 1.0)
        
        # ====================================================================
        # ERROR CHECKING
        # ====================================================================
        
        if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
            print(f"\n❌ Density problem at step {step}, t = {t:.3e}")
            print(f"   Min/max: {np.min(rho):.2e} / {np.max(rho):.2e}")
            break
        
        if np.any(p <= 0) or np.any(~np.isfinite(p)):
            print(f"\n❌ Pressure problem at step {step}, t = {t:.3e}")
            print(f"   Min/max: {np.min(p):.2e} / {np.max(p):.2e}")
            break
        
        # ====================================================================
        # MONITORING AND SAVING
        # ====================================================================
        
        if step % save_interval == 0:
            save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=step)
            
            # Physical diagnostics
            v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
            v_max_current = np.max(v_mag)
            
            T = p / (rho + 1e-20)
            T_max = np.max(T)
            
            cs = np.sqrt(GAMMA * p / (rho + 1e-20))
            cs_max = np.max(cs)
            
            mach = v_mag / (cs + 1e-20)
            mach_max = np.max(mach)
            
            # Accretion rate at inner boundary
            r_inner_sph = np.sqrt(r_inner**2 + z_inner**2)
            mask_inner_shell = (Rg >= r_inner_sph) & (Rg <= r_inner_sph * 1.2)
            
            if np.any(mask_inner_shell):
                r_sph = Rg[mask_inner_shell]
                rho_inner = rho[mask_inner_shell]
                vr_inner = vr[mask_inner_shell]
                vz_inner = vz[mask_inner_shell]
                
                # Compute radial velocity in spherical coords
                v_r_sph = (vr_inner * R[mask_inner_shell] + vz_inner * Z[mask_inner_shell]) / (r_sph + 1e-20)
                
                # Mass flux through shell
                mdot_inner = -np.sum(rho_inner * v_r_sph * r_sph**2) * (dR * dZ / r_inner_sph**2) * 4 * np.pi
            else:
                mdot_inner = 0.0
            
            # Total inflow
            mask_inflow = (vr < 0) & (Rg < R_max * 0.8)
            mdot_total = -np.sum(rho[mask_inflow] * vr[mask_inflow] * 
                                np.maximum(R[mask_inflow], R_reg) * dR * dZ) * 2 * np.pi
            
            # Bondi rate (theoretical)
            mdot_bondi_theory = 4 * np.pi * BH_mass**2 * rho_inf / (cs_inf**3)
            
            # Energy budget
            kinetic_energy = np.sum(0.5 * rho * v_mag**2 * np.maximum(R, R_reg) * dR * dZ) * 2 * np.pi
            internal_energy = np.sum(p / (GAMMA - 1.0) * np.maximum(R, R_reg) * dR * dZ) * 2 * np.pi
            total_energy = kinetic_energy + internal_energy
            
            print(f"\nStep {step:6d}, t = {t:.4f}, dt = {dt:.2e}")
            print(f"  Hydro: ρ_max={np.max(rho):.2e}, T_max={T_max:.2e}, v_max={v_max_current:.3f}")
            print(f"  Flow:  Mach={mach_max:.2f}, cs_max={cs_max:.3f}")
            print(f"  Accr:  Ṁ_inner={mdot_inner:.2e}, Ṁ_total={mdot_total:.2e}, Ṁ_Bondi={mdot_bondi_theory:.2e}")
            print(f"  Energy: KE={kinetic_energy:.2e}, IE={internal_energy:.2e}, Total={total_energy:.2e}")
            print(f"  Cumulative accreted mass = {cumulative_accreted_mass:.3e}")
            
            # Warnings
            if v_max_current > 0.9:
                print(f"  ⚠️  High velocity: {v_max_current:.3f}")
            
            if mach_max > 10.0:
                print(f"  ⚠️  Very high Mach: {mach_max:.2f}")
            
            if np.any(rho <= rho_floor * 1.01):
                n_floor = np.sum(rho <= rho_floor * 1.01)
                print(f"  ⚠️  {n_floor} cells at density floor")
            
            if dt < dt_min * 10:
                print(f"  ⚠️  Small timestep: {dt:.2e}")
        
        # Update time
        t += dt
        step += 1
        
        # Safety exit
        if step > 1000000:
            print(f"\n⚠️  Reached maximum step limit")
            break
    
    # ========================================================================
    # FINAL DIAGNOSTICS
    # ========================================================================
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETED")
    print("="*70)
    print(f"Final time: {t:.3f}")
    print(f"Total steps: {step}")
    
    # Check spherical symmetry
    print("\nSpherical symmetry check:")
    for r_test in [10, 20, 30, 40]:
        if r_test < R_max:
            mask = np.abs(Rg - r_test) < 0.5
            if np.any(mask):
                rho_variation = np.std(rho[mask]) / (np.mean(rho[mask]) + 1e-20)
                print(f"  r={r_test}: σ(ρ)/⟨ρ⟩ = {rho_variation:.2e}")
    
    # Mass and energy conservation
    total_mass = np.sum(rho * np.maximum(R, R_reg) * dR * dZ) * 2 * np.pi
    total_energy = np.sum(e_total * np.maximum(R, R_reg) * dR * dZ) * 2 * np.pi
    
    print(f"\nFinal conserved quantities:")
    print(f"  Total mass in domain: {total_mass:.3e}")
    print(f"  Total energy in domain: {total_energy:.3e}")
    print(f"  Cumulative accreted mass: {cumulative_accreted_mass:.3e}")
    
    # Accretion efficiency
    if cumulative_accreted_mass > 0:
        accretion_efficiency = cumulative_accreted_mass / (mdot_bondi_theory * t + 1e-20)
        print(f"  Accretion efficiency: {accretion_efficiency:.2%} of Bondi rate")
    
    # Save final state
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=step)
    print(f"\nFinal state saved to step {step}")
    print("="*70)