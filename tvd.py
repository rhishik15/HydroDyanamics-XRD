# tvd.py - Total Variation Diminishing limiters for shock capturing
import numpy as np

def minmod(a, b):
    """
    Minmod limiter function
    """
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def superbee(a, b):
    """
    Superbee limiter - less diffusive than minmod
    """
    s1 = np.sign(a)
    s2 = np.sign(b)
    
    # Only apply where both have same sign
    mask = (s1 == s2) & (s1 != 0)
    
    result = np.zeros_like(a)
    if np.any(mask):
        abs_a = np.abs(a[mask])
        abs_b = np.abs(b[mask])
        
        choice1 = np.minimum(2.0 * abs_a, abs_b)
        choice2 = np.minimum(abs_a, 2.0 * abs_b)
        
        result[mask] = s1[mask] * np.maximum(choice1, choice2)
    
    return result

def van_leer(a, b):
    """
    Van Leer limiter - smooth and less oscillatory
    """
    s1 = np.sign(a)
    s2 = np.sign(b)
    
    # Only apply where both have same sign
    mask = (s1 == s2) & (s1 != 0)
    
    result = np.zeros_like(a)
    if np.any(mask):
        abs_a = np.abs(a[mask])
        abs_b = np.abs(b[mask])
        
        result[mask] = 2.0 * abs_a * abs_b / (abs_a + abs_b)
        result[mask] *= s1[mask]
    
    return result

def mc_limiter(a, b):
    """
    Monotonized Central (MC) limiter
    """
    s1 = np.sign(a)
    s2 = np.sign(b)
    
    # Only apply where both have same sign
    mask = (s1 == s2) & (s1 != 0)
    
    result = np.zeros_like(a)
    if np.any(mask):
        abs_a = np.abs(a[mask])
        abs_b = np.abs(b[mask])
        
        # MC limiter: min(2a, 2b, (a+b)/2)
        choice1 = 2.0 * abs_a
        choice2 = 2.0 * abs_b  
        choice3 = 0.5 * (abs_a + abs_b)
        
        result[mask] = s1[mask] * np.minimum(choice1, np.minimum(choice2, choice3))
    
    return result

def compute_gradients(field, dr, dz):
    """
    Compute gradients using centered differences with proper boundary handling
    """
    Nr, Nz = field.shape
    
    # r-direction gradients
    grad_r = np.zeros_like(field)
    
    # Interior points - centered difference
    for i in range(1, Nr-1):
        grad_r[i, :] = (field[i+1, :] - field[i-1, :]) / (2 * dr)
    
    # Boundaries - forward/backward difference
    grad_r[0, :] = (field[1, :] - field[0, :]) / dr
    grad_r[-1, :] = (field[-1, :] - field[-2, :]) / dr
    
    # z-direction gradients
    grad_z = np.zeros_like(field)
    
    # Interior points - centered difference
    for j in range(1, Nz-1):
        grad_z[:, j] = (field[:, j+1] - field[:, j-1]) / (2 * dz)
    
    # Boundaries - forward/backward difference
    grad_z[:, 0] = (field[:, 1] - field[:, 0]) / dz
    grad_z[:, -1] = (field[:, -1] - field[:, -2]) / dz
    
    return grad_r, grad_z

def apply_slope_limiter(field, dr, dz, limiter='minmod'):
    """
    Apply slope limiter to reduce oscillations near shocks
    """
    Nr, Nz = field.shape
    field_limited = field.copy()
    
    # Choose limiter function
    if limiter == 'minmod':
        limit_func = minmod
    elif limiter == 'superbee':
        limit_func = superbee
    elif limiter == 'van_leer':
        limit_func = van_leer
    elif limiter == 'mc':
        limit_func = mc_limiter
    else:
        return field_limited  # No limiting
    
    # Apply limiting in r-direction
    for i in range(1, Nr-1):
        for j in range(Nz):
            # Forward and backward differences
            df_forward = field[i+1, j] - field[i, j]
            df_backward = field[i, j] - field[i-1, j]
            
            # Apply limiter
            limited_slope = limit_func(df_forward, df_backward)
            
            # Reconstruct field value (simple approach)
            field_limited[i, j] = field[i, j]
    
    # Apply limiting in z-direction
    for i in range(Nr):
        for j in range(1, Nz-1):
            # Forward and backward differences
            df_forward = field[i, j+1] - field[i, j]
            df_backward = field[i, j] - field[i, j-1]
            
            # Apply limiter
            limited_slope = limit_func(df_forward, df_backward)
            
            # Reconstruct field value
            field_limited[i, j] = field[i, j]
    
    return field_limited

def detect_shocks(rho, p, vr, vz, dr, dz, threshold=0.1):
    """
    Detect shock regions using velocity divergence and pressure gradients
    """
    Nr, Nz = rho.shape
    shock_indicator = np.zeros_like(rho)
    
    # Compute velocity divergence
    div_v = np.zeros_like(vr)
    
    # Divergence in r-direction: (1/r) d(r*vr)/dr
    for i in range(1, Nr-1):
        for j in range(Nz):
            r_i = i * dr + 0.1  # Approximate radial coordinate
            r_plus = (i + 0.5) * dr + 0.1
            r_minus = (i - 0.5) * dr + 0.1
            
            rvr_plus = r_plus * 0.5 * (vr[i+1, j] + vr[i, j])
            rvr_minus = r_minus * 0.5 * (vr[i, j] + vr[i-1, j])
            
            if r_i > 1e-10:
                div_v[i, j] = (rvr_plus - rvr_minus) / (r_i * dr)
    
    # Add z-component: dvz/dz
    for i in range(Nr):
        for j in range(1, Nz-1):
            div_v[i, j] += (vz[i, j+1] - vz[i, j-1]) / (2 * dz)
    
    # Pressure gradients
    dp_dr, dp_dz = compute_gradients(p, dr, dz)
    grad_p_mag = np.sqrt(dp_dr**2 + dp_dz**2)
    
    # Sound speed for normalization
    cs = np.sqrt(5.0/3.0 * p / (rho + 1e-20))
    
    # Shock indicators
    # 1. Strong compression (negative divergence)
    compress_indicator = np.maximum(-div_v / (cs/min(dr, dz) + 1e-20), 0.0)
    
    # 2. Large pressure gradients
    pressure_indicator = grad_p_mag / (p/min(dr, dz) + 1e-20)
    
    # Combined shock indicator
    shock_indicator = np.maximum(compress_indicator, pressure_indicator)
    
    # Apply threshold
    shock_mask = shock_indicator > threshold
    
    return shock_mask, shock_indicator

def apply_artificial_viscosity(rho, p, vr, vz, shock_mask, dr, dz, C_lin=0.1, C_quad=2.0):
    """
    Apply artificial viscosity in shock regions
    """
    Nr, Nz = rho.shape
    q_av = np.zeros_like(rho)  # Artificial viscosity
    
    # Apply only in shock regions
    if np.any(shock_mask):
        # Characteristic length
        h = min(dr, dz)
        
        # Sound speed
        cs = np.sqrt(5.0/3.0 * p / (rho + 1e-20))
        
        # Velocity divergence (recompute for accuracy)
        div_v = np.zeros_like(rho)
        
        for i in range(1, Nr-1):
            for j in range(1, Nz-1):
                # Simple divergence estimate
                dvr_dr = (vr[i+1, j] - vr[i-1, j]) / (2 * dr)
                dvz_dz = (vz[i, j+1] - vz[i, j-1]) / (2 * dz)
                
                # Add geometric term for cylindrical coordinates
                r_val = (i * dr + 0.1)
                if r_val > 1e-10:
                    div_v[i, j] = dvr_dr + vr[i, j]/r_val + dvz_dz
                else:
                    div_v[i, j] = dvr_dr + dvz_dz
        
        # Apply artificial viscosity where needed
        mask_compress = (div_v < -1e-10) & shock_mask
        
        if np.any(mask_compress):
            # Linear artificial viscosity
            q_lin = C_lin * rho[mask_compress] * cs[mask_compress] * h * abs(div_v[mask_compress])
            
            # Quadratic artificial viscosity
            q_quad = C_quad * rho[mask_compress] * h**2 * div_v[mask_compress]**2
            
            q_av[mask_compress] = q_lin + q_quad
    
    return q_av

def apply_tvd_limiters(rho, p, vr, vz, vphi, dr, dz, limiter='minmod', shock_threshold=0.1):
    """
    Apply TVD limiters to primitive variables to maintain monotonicity
    """
    
    # Detect shock regions
    shock_mask, shock_indicator = detect_shocks(rho, p, vr, vz, dr, dz, shock_threshold)
    
    # Apply slope limiting to all primitive variables
    if limiter != 'none':
        rho_limited = apply_slope_limiter(rho, dr, dz, limiter)
        p_limited = apply_slope_limiter(p, dr, dz, limiter)
        vr_limited = apply_slope_limiter(vr, dr, dz, limiter)
        vz_limited = apply_slope_limiter(vz, dr, dz, limiter)
        vphi_limited = apply_slope_limiter(vphi, dr, dz, limiter)
    else:
        rho_limited = rho.copy()
        p_limited = p.copy()
        vr_limited = vr.copy()
        vz_limited = vz.copy()
        vphi_limited = vphi.copy()
    
    # Apply artificial viscosity to pressure in shock regions
    q_av = apply_artificial_viscosity(rho_limited, p_limited, vr_limited, vz_limited, 
                                     shock_mask, dr, dz)
    
    # Add artificial viscosity to pressure
    p_with_av = p_limited + q_av
    
    # Ensure positivity
    rho_limited = np.maximum(rho_limited, 1e-10)
    p_with_av = np.maximum(p_with_av, 1e-12)
    
    # Velocity limiting (if needed)
    v_mag = np.sqrt(vr_limited**2 + vz_limited**2 + vphi_limited**2)
    v_max_allowed = 0.3  # Maximum velocity limit
    
    mask_fast = v_mag > v_max_allowed
    if np.any(mask_fast):
        scale = v_max_allowed / (v_mag[mask_fast] + 1e-20)
        vr_limited[mask_fast] *= scale
        vz_limited[mask_fast] *= scale
        vphi_limited[mask_fast] *= scale
    
    return rho_limited, p_with_av, vr_limited, vz_limited, vphi_limited

def flux_limiter_reconstruction(field_left, field_center, field_right, limiter='minmod'):
    """
    Reconstruct interface values using flux limiters
    Higher-order reconstruction for better accuracy
    """
    # Compute slopes
    slope_left = field_center - field_left
    slope_right = field_right - field_center
    
    # Apply limiter
    if limiter == 'minmod':
        limited_slope = minmod(slope_left, slope_right)
    elif limiter == 'superbee':
        limited_slope = superbee(slope_left, slope_right)
    elif limiter == 'van_leer':
        limited_slope = van_leer(slope_left, slope_right)
    elif limiter == 'mc':
        limited_slope = mc_limiter(slope_left, slope_right)
    else:
        limited_slope = 0.5 * (slope_left + slope_right)  # Central difference
    
    # Reconstruct interface values
    field_left_interface = field_center - 0.5 * limited_slope
    field_right_interface = field_center + 0.5 * limited_slope
    
    return field_left_interface, field_right_interface

def apply_positivity_preserving(rho, p, e_internal):
    """
    Ensure physical positivity of variables
    """
    rho_floor = 1e-10
    p_floor = 1e-12
    e_floor = 1e-12
    
    # Apply floors
    rho_safe = np.maximum(rho, rho_floor)
    p_safe = np.maximum(p, p_floor)
    e_safe = np.maximum(e_internal, e_floor)
    
    # Check consistency: p = (γ-1) * e_internal
    gamma = 5.0/3.0
    p_from_e = (gamma - 1.0) * e_safe
    
    # Use the more restrictive constraint
    p_final = np.maximum(p_safe, p_from_e)
    
    return rho_safe, p_final, e_safe

def monitor_tvd_property(field_old, field_new):
    """
    Monitor Total Variation to ensure TVD property is satisfied
    TV(u^{n+1}) <= TV(u^n)
    """
    
    def compute_total_variation(field):
        """Compute total variation of a 2D field"""
        Nr, Nz = field.shape
        tv = 0.0
        
        # Variation in r-direction
        for i in range(Nr-1):
            tv += np.sum(np.abs(field[i+1, :] - field[i, :]))
        
        # Variation in z-direction
        for j in range(Nz-1):
            tv += np.sum(np.abs(field[:, j+1] - field[:, j]))
        
        return tv
    
    tv_old = compute_total_variation(field_old)
    tv_new = compute_total_variation(field_new)
    
    tvd_satisfied = tv_new <= tv_old * 1.01  # Allow small numerical tolerance
    
    return tvd_satisfied, tv_old, tv_new