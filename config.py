# config.py - Improved for stability and physical realism

params = {
    # --- Grid resolution & domain (reduced for stability) ---
    "N_r": 900,           # reduced from 900 for initial testing
    "N_z": 900,           # reduced from 900 for initial testing  
    "R_max": 75.0,        # smaller domain for better resolution near BH
    "Z_max": 75.0,        # smaller domain for better resolution near BH

    # --- Physics ---
    "GAMMA": 5.0 / 3.0,   # adiabatic index (monoatomic gas)
    "r_inner": 2.5,       # inner absorption radius (> 2M)
    "r_outer": 5.0,      # outer boundary for sponge
    "BH_mass": 1.0,       # black hole mass in code units
    
    # --- Potential softening (more conservative) ---
    "pw_softening": 0.5,  # increased softening for stability
    "potential_cutoff": 2.0,  # minimum radius for potential calculation

    # --- Numerics (very conservative) ---
    "CFL": 0.1,           # much more conservative CFL
    "dt_max": 0.01,      # maximum allowed timestep
    "dt_min": 1e-8,       # minimum timestep before abort
    "t_end": 50.0,         # shorter simulation for testing
    "save_interval": 10,  # save less frequently
    
    # --- Floors (more aggressive) ---
    "rho_floor": 1e-8,    # higher density floor
    "p_floor": 1e-10,     # higher pressure floor  
    "e_floor": 1e-12,     # energy floor
    "v_max": 0.4,         # maximum allowed velocity (speed limiter)

    # --- Bondi inflow parameters ---
    "use_bondi_outer_bc": True,
    "bondi_mdot": 1.0,    # mass accretion rate
    "rho_inf": 1.0,       # asymptotic density
    "cs_inf": 0.1,        # asymptotic sound speed
    "bondi_radius": None, # will be computed as GM/cs_inf^2
    
    # --- Sponge/damping zones ---
    "sponge_inner_start": 3.0,   # inner sponge starts here
    "sponge_inner_width": 2.0,   # inner sponge thickness
    "sponge_outer_start": 5.0,  # outer sponge starts here
    "sponge_outer_width": 1.0,   # outer sponge thickness
    "sponge_strength": 0.1,      # damping strength
    "sponge_mode": "linear",     # "linear" or "exponential"
    
    # --- Velocity limiting ---
    "use_velocity_ceiling": True,
    "v_ceiling_factor": 0.5,     # limit to 0.5 * local sound speed
    "velocity_limiter_strength": 0.1,
}