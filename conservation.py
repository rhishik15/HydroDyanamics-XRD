# conservation.py - Hydrodynamic conservation equations in cylindrical coordinates
import numpy as np

def compute_fluxes(rho, p, vr, vz, vphi, R, dr, dz):
    """
    Compute flux divergences for all conservation equations
    using finite differences in cylindrical coordinates
    
    Conservation equations:
    1. ∂ρ/∂t + ∇·(ρv) = 0
    2. ∂(ρvᵣ)/∂t + ∇·(ρvᵣv + pδᵣ) = source terms
    3. ∂(ρvᵢ)/∂t + ∇·(ρvᵢv + pδᵢ) = source terms  
    4. ∂(ρvφ)/∂t + ∇·(ρvφv) = source terms
    5. ∂E/∂t + ∇·((E+p)v) = source terms
    """
    
    Nr, Nz = rho.shape
    
    # Initialize flux divergences
    dU1_dt = np.zeros_like(rho)  # Mass
    dU2_dt = np.zeros_like(rho)  # r-momentum
    dU3_dt = np.zeros_like(rho)  # z-momentum  
    dU4_dt = np.zeros_like(rho)  # φ-momentum
    dU5_dt = np.zeros_like(rho)  # Energy
    
    # Total energy
    e_total = 0.5 * rho * (vr**2 + vz**2 + vphi**2) + p / (5.0/3.0 - 1.0)
    
    # -------------------------
    # Radial fluxes (r-direction)
    # -------------------------
    for i in range(1, Nr-1):
        for j in range(Nz):
            # Interface positions
            r_plus = 0.5 * (R[i+1, j] + R[i, j])
            r_minus = 0.5 * (R[i, j] + R[i-1, j])
            
            # Mass flux: ρvᵣ
            F_mass_plus = 0.5 * (rho[i+1, j] * vr[i+1, j] + rho[i, j] * vr[i, j])
            F_mass_minus = 0.5 * (rho[i, j] * vr[i, j] + rho[i-1, j] * vr[i-1, j])
            
            # r-momentum flux: ρvᵣvᵣ + p
            F_rmom_plus = 0.5 * (rho[i+1, j] * vr[i+1, j]**2 + p[i+1, j] + 
                                 rho[i, j] * vr[i, j]**2 + p[i, j])
            F_rmom_minus = 0.5 * (rho[i, j] * vr[i, j]**2 + p[i, j] + 
                                  rho[i-1, j] * vr[i-1, j]**2 + p[i-1, j])
            
            # z-momentum flux: ρvᵣvᵢ  
            F_zmom_plus = 0.5 * (rho[i+1, j] * vr[i+1, j] * vz[i+1, j] + 
                                 rho[i, j] * vr[i, j] * vz[i, j])
            F_zmom_minus = 0.5 * (rho[i, j] * vr[i, j] * vz[i, j] + 
                                  rho[i-1, j] * vr[i-1, j] * vz[i-1, j])
            
            # φ-momentum flux: ρvᵣvφ
            F_phimom_plus = 0.5 * (rho[i+1, j] * vr[i+1, j] * vphi[i+1, j] + 
                                   rho[i, j] * vr[i, j] * vphi[i, j])
            F_phimom_minus = 0.5 * (rho[i, j] * vr[i, j] * vphi[i, j] + 
                                    rho[i-1, j] * vr[i-1, j] * vphi[i-1, j])
            
            # Energy flux: (E + p)vᵣ
            F_energy_plus = 0.5 * ((e_total[i+1, j] + p[i+1, j]) * vr[i+1, j] + 
                                   (e_total[i, j] + p[i, j]) * vr[i, j])
            F_energy_minus = 0.5 * ((e_total[i, j] + p[i, j]) * vr[i, j] + 
                                    (e_total[i-1, j] + p[i-1, j]) * vr[i-1, j])
            
            # Flux divergence in cylindrical coordinates: (1/r)∂(rF)/∂r
            r_center = R[i, j]
            if r_center > 1e-10:  # Avoid division by zero
                dU1_dt[i, j] -= (r_plus * F_mass_plus - r_minus * F_mass_minus) / (r_center * dr)
                dU2_dt[i, j] -= (r_plus * F_rmom_plus - r_minus * F_rmom_minus) / (r_center * dr)
                dU3_dt[i, j] -= (r_plus * F_zmom_plus - r_minus * F_zmom_minus) / (r_center * dr)
                dU4_dt[i, j] -= (r_plus * F_phimom_plus - r_minus * F_phimom_minus) / (r_center * dr)
                dU5_dt[i, j] -= (r_plus * F_energy_plus - r_minus * F_energy_minus) / (r_center * dr)
    
    # -------------------------
    # Vertical fluxes (z-direction)
    # -------------------------
    for i in range(Nr):
        for j in range(1, Nz-1):
            # Mass flux: ρvᵢ
            F_mass_plus = 0.5 * (rho[i, j+1] * vz[i, j+1] + rho[i, j] * vz[i, j])
            F_mass_minus = 0.5 * (rho[i, j] * vz[i, j] + rho[i, j-1] * vz[i, j-1])
            
            # r-momentum flux: ρvᵢvᵣ
            F_rmom_plus = 0.5 * (rho[i, j+1] * vz[i, j+1] * vr[i, j+1] + 
                                 rho[i, j] * vz[i, j] * vr[i, j])
            F_rmom_minus = 0.5 * (rho[i, j] * vz[i, j] * vr[i, j] + 
                                  rho[i, j-1] * vz[i, j-1] * vr[i, j-1])
            
            # z-momentum flux: ρvᵢvᵢ + p
            F_zmom_plus = 0.5 * (rho[i, j+1] * vz[i, j+1]**2 + p[i, j+1] + 
                                 rho[i, j] * vz[i, j]**2 + p[i, j])
            F_zmom_minus = 0.5 * (rho[i, j] * vz[i, j]**2 + p[i, j] + 
                                  rho[i, j-1] * vz[i, j-1]**2 + p[i, j-1])
            
            # φ-momentum flux: ρvᵢvφ
            F_phimom_plus = 0.5 * (rho[i, j+1] * vz[i, j+1] * vphi[i, j+1] + 
                                   rho[i, j] * vz[i, j] * vphi[i, j])
            F_phimom_minus = 0.5 * (rho[i, j] * vz[i, j] * vphi[i, j] + 
                                    rho[i, j-1] * vz[i, j-1] * vphi[i, j-1])
            
            # Energy flux: (E + p)vᵢ
            F_energy_plus = 0.5 * ((e_total[i, j+1] + p[i, j+1]) * vz[i, j+1] + 
                                   (e_total[i, j] + p[i, j]) * vz[i, j])
            F_energy_minus = 0.5 * ((e_total[i, j] + p[i, j]) * vz[i, j] + 
                                    (e_total[i, j-1] + p[i, j-1]) * vz[i, j-1])
            
            # Flux divergence: ∂F/∂z
            dU1_dt[i, j] -= (F_mass_plus - F_mass_minus) / dz
            dU2_dt[i, j] -= (F_rmom_plus - F_rmom_minus) / dz
            dU3_dt[i, j] -= (F_zmom_plus - F_zmom_minus) / dz
            dU4_dt[i, j] -= (F_phimom_plus - F_phimom_minus) / dz
            dU5_dt[i, j] -= (F_energy_plus - F_energy_minus) / dz
    
    # -------------------------
    # Geometric source terms for cylindrical coordinates
    # -------------------------
    for i in range(Nr):
        for j in range(Nz):
            r_val = R[i, j]
            if r_val > 1e-10:
                # Centrifugal force: ρvφ²/r (acts in r-direction)
                dU2_dt[i, j] += rho[i, j] * vphi[i, j]**2 / r_val
                
                # Pressure gradient geometric term
                dU2_dt[i, j] += p[i, j] / r_val  # This is already included in flux, but kept for clarity
    
    return dU1_dt, dU2_dt, dU3_dt, dU4_dt, dU5_dt


def compute_source_terms(rho, vr, vz, F_r, F_z, R):
    """
    Compute gravitational source terms
    """
    Nr, Nz = rho.shape
    
    # Initialize source terms
    S2 = np.zeros_like(rho)  # r-momentum source
    S3 = np.zeros_like(rho)  # z-momentum source  
    S5 = np.zeros_like(rho)  # Energy source
    
    # Gravitational force on momentum
    S2[:, :] = rho * F_r  # ρ * (-∂Φ/∂r)
    S3[:, :] = rho * F_z  # ρ * (-∂Φ/∂z)
    
    # Work done by gravitational force on energy
    S5[:, :] = rho * (vr * F_r + vz * F_z)  # ρv·F = ρv·(-∇Φ)
    
    return S2, S3, S5


def compute_pressure_gradient(p, dr, dz):
    """
    Compute pressure gradients using centered differences
    """
    Nr, Nz = p.shape
    
    dp_dr = np.zeros_like(p)
    dp_dz = np.zeros_like(p)
    
    # r-direction gradient
    for i in range(1, Nr-1):
        dp_dr[i, :] = (p[i+1, :] - p[i-1, :]) / (2 * dr)
    
    # Boundary extrapolation
    dp_dr[0, :] = (p[1, :] - p[0, :]) / dr
    dp_dr[-1, :] = (p[-1, :] - p[-2, :]) / dr
    
    # z-direction gradient  
    for j in range(1, Nz-1):
        dp_dz[:, j] = (p[:, j+1] - p[:, j-1]) / (2 * dz)
    
    # Boundary extrapolation
    dp_dz[:, 0] = (p[:, 1] - p[:, 0]) / dz
    dp_dz[:, -1] = (p[:, -1] - p[:, -2]) / dz
    
    return dp_dr, dp_dz


def compute_velocity_divergence(vr, vz, R, dr, dz):
    """
    Compute velocity divergence in cylindrical coordinates: ∇·v = (1/r)∂(rvr)/∂r + ∂vz/∂z
    """
    Nr, Nz = vr.shape
    div_v = np.zeros_like(vr)
    
    for i in range(1, Nr-1):
        for j in range(Nz):
            r_val = R[i, j]
            if r_val > 1e-10:
                # (1/r)∂(rvr)/∂r term
                rvr_plus = 0.5 * (R[i+1, j] * vr[i+1, j] + R[i, j] * vr[i, j])
                rvr_minus = 0.5 * (R[i, j] * vr[i, j] + R[i-1, j] * vr[i-1, j])
                
                div_r = (rvr_plus - rvr_minus) / (r_val * dr)
                div_v[i, j] = div_r
    
    # Add z-component ∂vz/∂z
    for i in range(Nr):
        for j in range(1, Nz-1):
            div_z = (vz[i, j+1] - vz[i, j-1]) / (2 * dz)
            div_v[i, j] += div_z
        
        # Boundary extrapolation for z-derivative
        div_v[i, 0] += (vz[i, 1] - vz[i, 0]) / dz
        div_v[i, -1] += (vz[i, -1] - vz[i, -2]) / dz
    
    return div_v


def apply_artificial_viscosity(rho, p, vr, vz, vphi, dr, dz, C_q=2.0):
    """
    Apply artificial viscosity for shock capturing
    Simple linear artificial viscosity
    """
    Nr, Nz = rho.shape
    
    # Compute velocity divergence
    div_v = compute_velocity_divergence(vr, vz, np.sqrt(np.arange(Nr)[:, None]**2), dr, dz)
    
    # Sound speed
    cs = np.sqrt(5.0/3.0 * p / (rho + 1e-20))
    
    # Artificial viscosity coefficient
    nu_av = np.zeros_like(rho)
    
    # Apply only where flow is compressing (div_v < 0)
    mask_compress = div_v < -1e-10
    if np.any(mask_compress):
        h = min(dr, dz)  # Characteristic length scale
        nu_av[mask_compress] = C_q * rho[mask_compress] * h**2 * abs(div_v[mask_compress])
    
    # Add artificial viscous pressure to real pressure
    p_with_av = p + nu_av
    
    return p_with_av, nu_av


def compute_timestep_constraint(rho, p, vr, vz, vphi, dr, dz, CFL=0.5):
    """
    Compute timestep constraints from various physical processes
    """
    # Sound speed
    cs = np.sqrt(np.maximum(5.0/3.0 * p / (rho + 1e-20), 1e-20))
    
    # Maximum signal speeds
    v_signal_r = np.abs(vr) + cs
    v_signal_z = np.abs(vz) + cs
    
    # CFL constraint
    dt_r = dr / (np.max(v_signal_r) + 1e-20)
    dt_z = dz / (np.max(v_signal_z) + 1e-20)
    dt_cfl = CFL * min(dt_r, dt_z)
    
    # Advection constraint
    v_max = np.max(np.sqrt(vr**2 + vz**2 + vphi**2))
    dt_advect = 0.5 * min(dr, dz) / (v_max + 1e-20)
    
    return min(dt_cfl, dt_advect)


def check_conservation(rho, vr, vz, vphi, e_total, R, dr, dz):
    """
    Check global conservation laws
    """
    # Total mass
    total_mass = np.sum(rho * R * dr * dz) * 2 * np.pi
    
    # Total momentum  
    total_momentum_r = np.sum(rho * vr * R * dr * dz) * 2 * np.pi
    total_momentum_z = np.sum(rho * vz * R * dr * dz) * 2 * np.pi
    total_momentum_phi = np.sum(rho * vphi * R * dr * dz) * 2 * np.pi
    
    # Total energy
    total_energy = np.sum(e_total * R * dr * dz) * 2 * np.pi
    
    return {
        'mass': total_mass,
        'momentum_r': total_momentum_r, 
        'momentum_z': total_momentum_z,
        'momentum_phi': total_momentum_phi,
        'energy': total_energy
    }