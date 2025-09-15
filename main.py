# main.py - Complete Hydrodynamic Black Hole Accretion Simulation
import numpy as np
from math import sqrt, log, exp
from config import params
from conservation import compute_fluxes, compute_source_terms
from tvd import apply_tvd_limiters
from save import save_all_enhanced

# -------------------------
# Grid & params  
# -------------------------
N_r, N_z = params["N_r"], params["N_z"]
R_max, Z_max = params["R_max"], params["Z_max"]
GAMMA = params["GAMMA"]

# Safety parameters
rho_floor = params.get("rho_floor", 1e-8)
p_floor = params.get("p_floor", 1e-10)
e_floor = params.get("e_floor", 1e-12)
v_max = params.get("v_max", 0.3)

# Physical parameters
BH_mass = params.get("BH_mass", 1.0)
rho_inf = params.get("rho_inf", 1.0)
cs_inf = params.get("cs_inf", 0.1)
r_inner = params.get("r_inner", 2.5)

# Compute Bondi radius
R_bondi = BH_mass / (cs_inf**2)
print(f"Bondi radius: {R_bondi:.3f}")

# Time stepping parameters
CFL = params.get("CFL", 0.1)
dt_max = params.get("dt_max", 0.001)
dt_min = params.get("dt_min", 1e-8)
t_end = params.get("t_end", 5.0)
save_interval = params.get("save_interval", 10)

# -------------------------
# Grid setup - Centered black hole
# -------------------------
r_min_grid = 0.1  # Small positive value to avoid r=0
r = np.linspace(-R_max, R_max, N_r)
z = np.linspace(-Z_max, Z_max, N_z)

dr, dz = r[1] - r[0], z[1] - z[0]
R, Z = np.meshgrid(r, z, indexing="ij")

# Spherical radius from black hole center
Rg = np.sqrt(R**2 + Z**2)

print(f"Grid spacing: dr = {dr:.3f}, dz = {dz:.3f}")

# -------------------------
# Bondi solution for boundary conditions
# -------------------------
def bondi_solution_3d(r_val, z_val):
    """
    Compute 3D Bondi solution at given cylindrical coordinates
    """
    R_sph = sqrt(r_val**2 + z_val**2)
    R_sph = max(R_sph, r_inner)
    
    # Bondi parameters
    r_sonic = 0.25 * BH_mass / (cs_inf**2)  # Sonic radius
    
    if R_sph > 3.0 * r_sonic:
        # Subsonic region - gentle inflow
        rho = rho_inf * (1.0 + 0.5 * (r_sonic / R_sph))
        v_r_sph = -cs_inf * (r_sonic / R_sph)
    else:
        # Transition to supersonic
        rho = rho_inf * (r_sonic / R_sph)**0.5
        v_r_sph = -sqrt(BH_mass / R_sph) * 0.5  # Limited velocity
    
    # Convert spherical radial velocity to cylindrical components
    if R_sph > 1e-10:
        cos_theta = r_val / R_sph  # cos(theta) in spherical coords
        sin_theta = abs(z_val) / R_sph  # |sin(theta)|
        
        v_r_cyl = v_r_sph * cos_theta
        v_z_cyl = v_r_sph * sin_theta * np.sign(z_val)
    else:
        v_r_cyl = 0.0
        v_z_cyl = 0.0
    
    # Pressure from ideal gas
    p = cs_inf**2 * rho / GAMMA
    
    return rho, p, v_r_cyl, v_z_cyl

# -------------------------
# Black hole potential (Paczy??ski-Wiita)
# -------------------------
def compute_potential_and_forces():
    """
    Compute gravitational potential and force components
    """
    # Softened radius to avoid singularity
    softening = params.get("pw_softening", 0.5)
    r_g = 2.0 * BH_mass  # Schwarzschild radius
    
    r_soft = np.maximum(Rg, softening)
    denom = r_soft - 0.5 * r_g + softening
    denom = np.maximum(denom, softening)
    
    # Potential
    Phi = -BH_mass / denom
    
    # Force magnitude
    F_mag = BH_mass / (denom**2)
    
    # Force components in cylindrical coordinates
    R_safe = np.maximum(r_soft, 1e-10)
    F_r = -F_mag * R / R_safe  # Radial component
    F_z = -F_mag * Z / R_safe  # Vertical component
    
    return Phi, F_r, F_z

# -------------------------
# Initialize flow field
# -------------------------
def initialize_bondi_flow():
    """
    Initialize with spherically symmetric Bondi inflow
    """
    rho = np.zeros((N_r, N_z))
    p = np.zeros((N_r, N_z))
    vr = np.zeros((N_r, N_z))
    vz = np.zeros((N_r, N_z))
    vphi = np.zeros((N_r, N_z))  # No azimuthal flow initially
    
    print("Initializing Bondi flow...")
    
    for i in range(N_r):
        for j in range(N_z):
            r_val, z_val = R[i, j], Z[i, j]
            
            rho_val, p_val, vr_val, vz_val = bondi_solution_3d(r_val, z_val)
            
            # Apply floors and limits
            rho[i, j] = max(rho_val, rho_floor)
            p[i, j] = max(p_val, p_floor)
            vr[i, j] = np.clip(vr_val, -v_max, v_max)
            vz[i, j] = np.clip(vz_val, -v_max, v_max)
            vphi[i, j] = 0.0
    
    # Total energy density
    kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
    internal = p / (GAMMA - 1.0)
    e_total = kinetic + internal
    
    print(f"Initial conditions: rho range [{np.min(rho):.2e}, {np.max(rho):.2e}]")
    print(f"Initial conditions: p range [{np.min(p):.2e}, {np.max(p):.2e}]")
    print(f"Initial conditions: v_max = {np.max(np.sqrt(vr**2 + vz**2)):.3f}")
    
    return rho, p, vr, vz, vphi, e_total

# -------------------------
# Boundary conditions
# -------------------------
def apply_boundary_conditions(rho, p, vr, vz, vphi):
    """
    Apply boundary conditions:
    1. Inner absorption zone
    2. Outer Bondi inflow
    3. Axisymmetric conditions on z-boundaries
    """
    
    # 1. Inner absorption (inside r_inner)
    mask_inner = (Rg < r_inner)
    if np.any(mask_inner):
        rho[mask_inner] = rho_floor
        p[mask_inner] = p_floor
        vr[mask_inner] = 0.0
        vz[mask_inner] = 0.0
        vphi[mask_inner] = 0.0
    
    # 2. Outer boundary - impose Bondi inflow
    r_outer = params.get("r_outer", R_max * 0.9)
    mask_outer = (Rg > r_outer)
    
    if np.any(mask_outer):
        for i in range(N_r):
            for j in range(N_z):
                if mask_outer[i, j]:
                    r_val, z_val = R[i, j], Z[i, j]
                    rho_b, p_b, vr_b, vz_b = bondi_solution_3d(r_val, z_val)
                    
                    # Relax toward Bondi solution
                    alpha = 0.1  # Relaxation strength
                    rho[i, j] = (1 - alpha) * rho[i, j] + alpha * rho_b
                    p[i, j] = (1 - alpha) * p[i, j] + alpha * p_b
                    vr[i, j] = (1 - alpha) * vr[i, j] + alpha * vr_b
                    vz[i, j] = (1 - alpha) * vz[i, j] + alpha * vz_b
    
    # 3. Axisymmetric boundaries (z = ±Z_max)
    # Symmetric quantities
    rho[:, 0] = rho[:, 1]
    rho[:, -1] = rho[:, -2]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    vr[:, 0] = vr[:, 1]
    vr[:, -1] = vr[:, -2]
    vphi[:, 0] = vphi[:, 1]
    vphi[:, -1] = vphi[:, -2]
    
    # Antisymmetric quantity (z-velocity)
    vz[:, 0] = -vz[:, 1]
    vz[:, -1] = -vz[:, -2]
    
    # 4. r = 0 axis (if included)
    if r[0] < 1e-6:
        rho[0, :] = rho[1, :]
        p[0, :] = p[1, :]
        vr[0, :] = 0.0  # No radial flow on axis
        vz[0, :] = vz[1, :]
        vphi[0, :] = 0.0  # No azimuthal flow on axis
    
    # 5. Final safety floors
    rho[:] = np.maximum(rho, rho_floor)
    p[:] = np.maximum(p, p_floor)
    
    # Velocity limiting
    v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
    mask_fast = v_mag > v_max
    if np.any(mask_fast):
        scale_factor = v_max / (v_mag[mask_fast] + 1e-20)
        vr[mask_fast] *= scale_factor
        vz[mask_fast] *= scale_factor
        vphi[mask_fast] *= scale_factor

# -------------------------
# Adaptive timestep
# -------------------------
def compute_timestep(rho, p, vr, vz, vphi):
    """
    Compute stable timestep based on CFL condition
    """
    # Sound speed
    cs = np.sqrt(np.maximum(GAMMA * p / (rho + 1e-20), 1e-20))
    
    # Maximum signal speeds
    sr_max = np.max(np.abs(vr) + cs)
    sz_max = np.max(np.abs(vz) + cs)
    
    # CFL timestep
    dt_cfl = CFL * min(dr / max(sr_max, 1e-10), 
                       dz / max(sz_max, 1e-10))
    
    # Additional constraint from maximum velocity
    v_max_grid = np.max(np.sqrt(vr**2 + vz**2 + vphi**2))
    dt_vel = 0.5 * min(dr, dz) / max(v_max_grid, 1e-10)
    
    # Final timestep
    dt = min(dt_cfl, dt_vel, dt_max)
    dt = max(dt, dt_min)
    
    return dt

# -------------------------
# Conservative variable conversion
# -------------------------
def primitive_to_conservative(rho, vr, vz, vphi, e_total):
    """Convert primitive to conservative variables"""
    U1 = rho.copy()                    # Mass density
    U2 = rho * vr                      # r-momentum density  
    U3 = rho * vz                      # z-momentum density
    U4 = rho * vphi                    # φ-momentum density
    U5 = e_total.copy()                # Total energy density
    
    return U1, U2, U3, U4, U5

def conservative_to_primitive(U1, U2, U3, U4, U5):
    """Convert conservative to primitive variables"""
    rho = np.maximum(U1, rho_floor)
    
    # Velocities (with safety for low density)
    vr = np.divide(U2, rho, out=np.zeros_like(U2), where=(rho > rho_floor))
    vz = np.divide(U3, rho, out=np.zeros_like(U3), where=(rho > rho_floor))
    vphi = np.divide(U4, rho, out=np.zeros_like(U4), where=(rho > rho_floor))
    
    # Energy and pressure
    e_total = np.maximum(U5, e_floor)
    kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
    internal = e_total - kinetic
    internal = np.maximum(internal, e_floor)
    p = (GAMMA - 1.0) * internal
    p = np.maximum(p, p_floor)
    
    return rho, vr, vz, vphi, p, e_total

# -------------------------
# Main simulation
# -------------------------
if __name__ == "__main__":
    print("="*60)
    print("HYDRODYNAMIC BLACK HOLE ACCRETION SIMULATION")
    print("="*60)
    print(f"Grid: {N_r} × {N_z}")
    print(f"Domain: r ∈ [{-R_max}, {R_max}], z ∈ [{-Z_max}, {Z_max}]")
    print(f"Bondi radius: {R_bondi:.3f}")
    print(f"Black hole mass: {BH_mass}")
    print(f"CFL number: {CFL}")
    print("="*60)
    
    # Initialize
    rho, p, vr, vz, vphi, e_total = initialize_bondi_flow()
    
    # Compute gravitational forces
    Phi, F_r, F_z = compute_potential_and_forces()
    print(f"Potential range: [{np.min(Phi):.2e}, {np.max(Phi):.2e}]")
    
    # Apply initial boundary conditions
    apply_boundary_conditions(rho, p, vr, vz, vphi)
    
    # Recompute total energy after boundary application
    kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
    internal = p / (GAMMA - 1.0)
    e_total = kinetic + internal
    
    # Save initial state
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0)
    print("Initial state saved")
    
    # Time evolution
    t = 0.0
    step = 0
    
    while t < t_end:
        # Compute timestep
        dt = compute_timestep(rho, p, vr, vz, vphi)
        
        if t + dt > t_end:
            dt = t_end - t
        
        # Store old state
        rho_old = rho.copy()
        p_old = p.copy()
        vr_old = vr.copy()
        vz_old = vz.copy()
        vphi_old = vphi.copy()
        e_total_old = e_total.copy()
        
        # Convert to conservative variables
        U1, U2, U3, U4, U5 = primitive_to_conservative(rho, vr, vz, vphi, e_total)
        
        # RK2 Step 1: Half timestep
        dU1_dt, dU2_dt, dU3_dt, dU4_dt, dU5_dt = compute_fluxes(
            rho, p, vr, vz, vphi, R, dr, dz)
        
        # Add source terms (gravity)
        S2, S3, S5 = compute_source_terms(rho, vr, vz, F_r, F_z, R)
        dU2_dt += S2
        dU3_dt += S3  
        dU5_dt += S5
        
        # Intermediate state
        U1_half = U1 + 0.5 * dt * dU1_dt
        U2_half = U2 + 0.5 * dt * dU2_dt
        U3_half = U3 + 0.5 * dt * dU3_dt
        U4_half = U4 + 0.5 * dt * dU4_dt
        U5_half = U5 + 0.5 * dt * dU5_dt
        
        # Convert back to primitive
        rho_half, vr_half, vz_half, vphi_half, p_half, e_half = \
            conservative_to_primitive(U1_half, U2_half, U3_half, U4_half, U5_half)
        
        # Apply TVD limiters
        rho_half, p_half, vr_half, vz_half, vphi_half = apply_tvd_limiters(
            rho_half, p_half, vr_half, vz_half, vphi_half, dr, dz)
        
        # Apply boundaries to intermediate state
        apply_boundary_conditions(rho_half, p_half, vr_half, vz_half, vphi_half)
        
        # RK2 Step 2: Full timestep using intermediate state
        dU1_dt2, dU2_dt2, dU3_dt2, dU4_dt2, dU5_dt2 = compute_fluxes(
            rho_half, p_half, vr_half, vz_half, vphi_half, R, dr, dz)
        
        # Source terms at half step
        S2_half, S3_half, S5_half = compute_source_terms(
            rho_half, vr_half, vz_half, F_r, F_z, R)
        dU2_dt2 += S2_half
        dU3_dt2 += S3_half
        dU5_dt2 += S5_half
        
        # Final update
        U1_new = U1 + dt * dU1_dt2
        U2_new = U2 + dt * dU2_dt2
        U3_new = U3 + dt * dU3_dt2
        U4_new = U4 + dt * dU4_dt2
        U5_new = U5 + dt * dU5_dt2
        
        # Convert back to primitive variables
        rho, vr, vz, vphi, p, e_total = conservative_to_primitive(
            U1_new, U2_new, U3_new, U4_new, U5_new)
        
        # Apply TVD limiters to final state
        rho, p, vr, vz, vphi = apply_tvd_limiters(rho, p, vr, vz, vphi, dr, dz)
        
        # Final boundary conditions
        apply_boundary_conditions(rho, p, vr, vz, vphi)
        
        # Update total energy consistently
        kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
        internal = p / (GAMMA - 1.0)
        e_total = kinetic + internal
        
        # Check for problems
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
            print(f"Density problem at step {step}, t = {t:.3e}")
            print(f"Min/max density: {np.min(rho):.2e} / {np.max(rho):.2e}")
            break
        
        if np.any(~np.isfinite(p)) or np.any(p <= 0):
            print(f"Pressure problem at step {step}, t = {t:.3e}")
            print(f"Min/max pressure: {np.min(p):.2e} / {np.max(p):.2e}")
            break
        
        # Monitor and save
        if step % save_interval == 0:
            save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=step)
            
            v_max_current = np.max(np.sqrt(vr**2 + vz**2 + vphi**2))
            mdot = np.sum(rho * vr * R * dr * dz) * 2 * np.pi  # Accretion rate
            
            print(f"Step {step:6d}, t = {t:.4f}, dt = {dt:.2e}, "
                  f"max(ρ) = {np.max(rho):.2e}, max(v) = {v_max_current:.3f}, "
                  f"Ṁ = {mdot:.2e}")
        
        t += dt
        step += 1
        
        # Safety exit
        if step > 100000:
            print("Maximum steps reached")
            break
    
    print(f"\nSimulation completed!")
    print(f"Final time: {t:.3f}")
    print(f"Total steps: {step}")
    print(f"Final accretion rate: {np.sum(rho * vr * R * dr * dz) * 2 * np.pi:.2e}")
    
    # Save final state
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=step)