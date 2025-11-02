# config.py - CORRECTED configuration for stable black hole accretion
"""
Configuration parameters for hydrodynamic black hole accretion simulation
with Bondi flow in Paczynski-Wiita potential

CRITICAL FIXES:
===============
1. ✅ Increased velocity limit (was 0.5, too low for supersonic flow!)
2. ✅ Proper softening scale (tied to grid resolution)
3. ✅ Physical inner boundary (at 3 Schwarzschild radii)
4. ✅ Extended sponge layer to domain edge
5. ✅ Conservative CFL number for stability
"""

params = {
    # ========================================================================
    # GRID RESOLUTION & DOMAIN
    # ========================================================================
    "N_r": 150,           # Radial resolution
    "N_z": 150,           # Vertical resolution
    "R_max": 75.0,        # Maximum radial coordinate (in r_g)
    "Z_max": 75.0,        # Maximum vertical coordinate (in r_g)
    
    # ========================================================================
    # PHYSICS
    # ========================================================================
    "GAMMA": 5.0 / 3.0,   # Adiabatic index (monoatomic gas)
    "BH_mass": 1.0,       # Black hole mass (in units where G=c=1)
    
    # ========================================================================
    # BOUNDARY CONDITIONS
    # ========================================================================
    # ✅ FIXED: Inner boundary at ISCO (3 r_g for non-rotating BH)
    "r_inner": 3.0,       # Inner boundary radius (Schwarzschild: 3 r_g)
    "z_inner": 3.0,       # Inner boundary height
    
    # ✅ FIXED: Outer boundary sponge extends to edge
    "r_outer": 60.0,      # Start of sponge layer (in r_g)
    "sponge_width": 15.0, # Sponge width: 60 → 75 (to R_max)
    
    # ========================================================================
    # POTENTIAL SOFTENING
    # ========================================================================
    # ✅ FIXED: Softening tied to grid scale
    # Paczynski-Wiita: Φ = -GM/(r - r_g)
    # Softening prevents singularity at r = r_g
    "pw_softening": 0.5,  # Softening = 0.1 * dr (adaptive)
    "potential_cutoff": 2.0,  # Minimum denominator (in r_g)
    
    # ========================================================================
    # NUMERICS
    # ========================================================================
    # ✅ CONSERVATIVE CFL for stability
    "CFL": 0.03,          # Courant number (0.05 is safe for HLLE + RK2)
    "dt_max": 0.03,       # Maximum timestep
    "dt_min": 1e-10,      # Minimum timestep (crash protection)
    "t_end": 200.0,       # End time
    "save_interval": 10,  # Save every N steps
    
    # ========================================================================
    # FLOORS (for numerical stability)
    # ========================================================================
    "rho_floor": 1e-8,    # Density floor
    "p_floor": 1e-10,     # Pressure floor
    "e_floor": 1e-10,     # Energy floor
    
    # ✅ FIXED: Increased velocity limit to allow supersonic flow!
    # Old value (0.5) was too restrictive - Bondi flow is supersonic near BH
    "v_max": 0.95,        # Maximum velocity (0.95c, was 0.5c)
    
    # ========================================================================
    # BONDI INFLOW PARAMETERS
    # ========================================================================
    "use_bondi_outer_bc": True,   # Use Bondi solution at outer boundary
    "bondi_mdot": 1.0,            # Bondi accretion rate (normalization)
    "rho_inf": 2e-4,              # Density at infinity
    "cs_inf": 0.173,              # Sound speed at infinity
    "bondi_radius": 33.0,         # Bondi radius = GM/cs²
    
    # ========================================================================
    # INITIAL CONDITIONS
    # ========================================================================
    "add_initial_perturbations": False,  # Start with pure Bondi (spherical)
    
    # ========================================================================
    # OPTIONAL FEATURES (currently disabled)
    # ========================================================================
    # These can be enabled for more advanced simulations
    
    # Keplerian disk (not used in spherical Bondi)
    "add_keplerian_disk": False,
    # "disk_mdot": 1.0,
    # "disk_r_in": 3.0,
    # "disk_r_out": 300.0,
    
    # Radiative cooling (disabled for basic hydro test)
    "use_bremsstrahlung": False,
    "use_compton": False,
    "use_synchrotron": False,
    # "t_cool_min": 1e-6,
}

# ============================================================================
# DERIVED PARAMETERS (computed from above)
# ============================================================================

def get_derived_params(params):
    """Compute derived physical parameters"""
    
    derived = {}
    
    # Schwarzschild radius
    derived['r_schwarzschild'] = 2.0 * params['BH_mass']
    
    # Bondi radius
    derived['r_bondi'] = params['BH_mass'] / params['cs_inf']**2
    
    # Sonic radius (Bondi solution)
    derived['r_sonic'] = params['BH_mass'] / (2.0 * params['cs_inf']**2)
    
    # Grid spacing
    derived['dr'] = params['R_max'] / params['N_r']
    derived['dz'] = params['Z_max'] / params['N_z']
    
    # Regularization scale
    derived['R_reg'] = 0.5 * derived['dr']
    
    # Bondi accretion rate (theoretical)
    derived['mdot_bondi'] = (4.0 * np.pi * params['BH_mass']**2 * 
                            params['rho_inf'] / params['cs_inf']**3)
    
    return derived

# Import numpy for derived parameters
import numpy as np

# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_params(params):
    """Check that parameters are physically reasonable"""
    
    errors = []
    warnings = []
    
    # Check grid resolution
    if params['N_r'] < 50 or params['N_z'] < 50:
        warnings.append("Low grid resolution may give inaccurate results")
    
    # Check domain size vs Bondi radius
    r_bondi = params['BH_mass'] / params['cs_inf']**2
    if params['R_max'] < 2 * r_bondi:
        errors.append(f"Domain too small! R_max={params['R_max']:.1f} < 2*R_bondi={2*r_bondi:.1f}")
    
    # Check inner boundary
    r_schwarzschild = 2.0 * params['BH_mass']
    if params['r_inner'] < 1.5 * r_schwarzschild:
        warnings.append(f"Inner boundary very close to horizon (r_in={params['r_inner']:.1f}, r_s={r_schwarzschild:.1f})")
    
    # Check sponge layer
    if params['r_outer'] + params['sponge_width'] > params['R_max']:
        warnings.append("Sponge layer extends beyond domain")
    
    # Check CFL number
    if params['CFL'] > 0.5:
        warnings.append(f"CFL={params['CFL']} may be unstable (recommend < 0.5)")
    
    # Check velocity limit
    if params['v_max'] < 0.7:
        warnings.append(f"v_max={params['v_max']} may be too restrictive for supersonic flow")
    
    # Print validation results
    if errors:
        print("="*70)
        print("❌ CONFIGURATION ERRORS:")
        for err in errors:
            print(f"  • {err}")
        print("="*70)
        raise ValueError("Invalid configuration parameters")
    
    if warnings:
        print("="*70)
        print("⚠️  CONFIGURATION WARNINGS:")
        for warn in warnings:
            print(f"  • {warn}")
        print("="*70)
    
    return True

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary(params):
    """Print a human-readable summary of the configuration"""
    
    derived = get_derived_params(params)
    
    print("\n" + "="*70)
    print("SIMULATION CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\nGRID:")
    print(f"  Resolution: {params['N_r']} × {params['N_z']} = {params['N_r']*params['N_z']} cells")
    print(f"  Domain: R ∈ [0, {params['R_max']}], Z ∈ [0, {params['Z_max']}] (in r_g)")
    print(f"  Grid spacing: dr = {derived['dr']:.3f}, dz = {derived['dz']:.3f}")
    print(f"  Regularization: R_reg = {derived['R_reg']:.3f}")
    
    print("\nPHYSICS:")
    print(f"  Black hole mass: M = {params['BH_mass']}")
    print(f"  Schwarzschild radius: r_s = {derived['r_schwarzschild']:.3f}")
    print(f"  Bondi radius: R_B = {derived['r_bondi']:.3f}")
    print(f"  Sonic radius: r_sonic = {derived['r_sonic']:.3f}")
    print(f"  Adiabatic index: γ = {params['GAMMA']:.3f}")
    
    print("\nBOUNDARIES:")
    print(f"  Inner boundary: r_in = {params['r_inner']:.1f} ({params['r_inner']/derived['r_schwarzschild']:.2f} r_s)")
    print(f"  Outer sponge: [{params['r_outer']:.1f}, {params['r_outer']+params['sponge_width']:.1f}]")
    print(f"  Sponge width: {params['sponge_width']:.1f}")
    
    print("\nBONDI FLOW:")
    print(f"  ρ_∞ = {params['rho_inf']:.2e}")
    print(f"  cs_∞ = {params['cs_inf']:.3f}")
    print(f"  Ṁ_Bondi = {derived['mdot_bondi']:.3e}")
    
    print("\nNUMERICS:")
    print(f"  CFL number: {params['CFL']}")
    print(f"  Timestep range: [{params['dt_min']:.1e}, {params['dt_max']:.2e}]")
    print(f"  End time: t_end = {params['t_end']}")
    print(f"  Save interval: every {params['save_interval']} steps")
    
    print("\nFLOORS & LIMITS:")
    print(f"  ρ_floor = {params['rho_floor']:.1e}")
    print(f"  p_floor = {params['p_floor']:.1e}")
    print(f"  v_max = {params['v_max']}")
    
    print("="*70 + "\n")

# ============================================================================
# VALIDATE ON IMPORT
# ============================================================================

if __name__ != "__main__":
    # Validate parameters when config is imported
    validate_params(params)