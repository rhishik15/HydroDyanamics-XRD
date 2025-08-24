#!/usr/bin/env python3
# init_state.py — XRB corona setup (nonrelativistic, BH + Bondi inflow)
"""
Initial conditions and boundary helpers for the 2D hydro solver
simulating an X-ray binary hot corona around a BH.

Key points:
 - Paczyński–Wiita potential models BH gravity.
 - Background = hot dilute corona.
 - Boundaries inject Bondi-like inflow of hot plasma.
 - Sound speeds limited (cs <= CS_SAFE) to stay Newtonian.
 - Absorber region inside Rs acts as sink for matter.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from conservation_normal import np_paczynski_wiita_potential

# ---------------- Defaults ----------------
N_r = 801
N_z = 801
R_max = 100.0
Z_max = 100.0
GAMMA = 5.0 / 3.0

CFL = 0.35
MIN_INTERNAL = 1e-12
CS_SAFE = 0.3   # sound speed cap (code units)

# ---------------- Grid builder ----------------
def make_grid(nr: int = N_r, nz: int = N_z, Rmax: float = R_max, Zmax: float = None):
    if Zmax is None:
        Zmax = Rmax
    r = np.linspace(-Rmax, Rmax, nr)
    z = np.linspace(-Zmax, Zmax, nz)
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    R_grid, Z_grid = np.meshgrid(r, z, indexing="ij")
    return r, z, dr, dz, R_grid, Z_grid

# ---------------- Initial conditions ----------------
def initial_conditions(R_grid: np.ndarray, Z_grid: np.ndarray):
    nr, nz = R_grid.shape

    # Bondi-like inflow params (scaled)
    v_in = 0.02
    vphi_b = 0.0
    rho_inj_amp = 1e-2
    T0 = 0.1

    # Radial dependence for inflow
    R_abs = np.abs(R_grid)
    R_max_grid = float(np.max(R_abs)) if np.max(R_abs) > 0 else 1.0
    rho_inj_profile = rho_inj_amp * (1.0 - 0.7 * (R_abs / R_max_grid))
    T_profile = T0 * (1.0 - 0.5 * (R_abs / R_max_grid))
    p_inj_profile = rho_inj_profile * T_profile

    rho_inj = float(np.mean(rho_inj_profile))
    p_inj = float(np.mean(p_inj_profile))

    # Hot corona background
    rho_bg = 0.3
    p_bg = 0.01

    # Smooth taper from boundary → interior
    smoothing_width = max(4, int(min(nr, nz) * 0.02))
    r_idx = np.arange(nr)[:, None]
    z_idx = np.arange(nz)[None, :]
    dist = np.minimum(np.minimum(r_idx, nr - 1 - r_idx), np.minimum(z_idx, nz - 1 - z_idx))
    taper = np.exp(-dist.astype(np.float64) / float(smoothing_width))

    # Blended fields
    p = p_bg + (p_inj - p_bg) * taper
    rho = rho_bg + (rho_inj - rho_bg) * taper

    # Velocities
    vr = np.zeros_like(rho)
    vz = np.zeros_like(rho)
    vphi = np.full(rho.shape, vphi_b)

    # Boundary strips set to inflow
    rho[0, :] = rho_inj_profile[0, :]
    p[0, :] = p_inj_profile[0, :]
    rho[-1, :] = rho_inj_profile[-1, :]
    p[-1, :] = p_inj_profile[-1, :]
    rho[:, 0] = rho_inj_profile[:, 0]
    p[:, 0] = p_inj_profile[:, 0]
    rho[:, -1] = rho_inj_profile[:, -1]
    p[:, -1] = p_inj_profile[:, -1]

    # Inward velocities
    vr[0, :] = +abs(v_in)
    vr[-1, :] = -abs(v_in)
    vz[:, 0] = +abs(v_in)
    vz[:, -1] = -abs(v_in)

    # Conserved vars
    mom_r = rho * vr
    mom_z = rho * vz
    mom_phi = rho * vphi
    e_total = p / (GAMMA - 1.0) + 0.5 * rho * (vr**2 + vz**2 + vphi**2)

    # Clamp cs <= CS_SAFE
    internal = p / (GAMMA - 1.0)
    denom = GAMMA * (GAMMA - 1.0)
    internal_max = rho * (CS_SAFE**2) / denom
    mask = internal > internal_max
    if np.any(mask):
        internal[mask] = internal_max[mask]
        p = (GAMMA - 1.0) * internal
        e_total = internal + 0.5 * rho * (vr**2 + vz**2 + vphi**2)
        print(f"[WARN] init: clamped internal in {np.count_nonzero(mask)} cell(s) for cs<= {CS_SAFE:.2f}")

    rho = np.nan_to_num(rho, nan=1e-8, posinf=1e8)
    p = np.nan_to_num(p, nan=MIN_INTERNAL, posinf=1e8)
    e_total = np.nan_to_num(e_total, nan=MIN_INTERNAL, posinf=1e8)

    params = dict(
        rho_inj_profile=rho_inj_profile,
        p_inj_profile=p_inj_profile,
        rho_in=rho_inj_profile, p_in=p_inj_profile,
        rho_inj=rho_inj, p_inj=p_inj,
        v_in=v_in, vphi_b=vphi_b,
        rho_bg=rho_bg, p_bg=p_bg,
        smoothing_width=smoothing_width,
        gamma=GAMMA,
    )

    cs = np.sqrt(GAMMA * p / np.maximum(rho, 1e-20))
    print(f"[DIAG] init: rho[min,max]=[{rho.min():.2e},{rho.max():.2e}], p[min,max]=[{p.min():.2e},{p.max():.2e}], cs_max={cs.max():.2e}")

    return rho, p, vr, vz, vphi, mom_r, mom_z, mom_phi, e_total, params

# ---------------- Boundary conditions ----------------
def apply_boundaries(rho, mom_r, mom_z, mom_phi, p, e_total, R_grid, Z_grid, params: Dict):
    gamma = params.get('gamma', GAMMA)
    v_in = params.get('v_in', 0.0)
    vphi_b = params.get('vphi_b', 0.0)
    rho_in_profile = params.get('rho_in', None)
    p_in_profile = params.get('p_in', None)

    def set_face(idx, vr_b, vz_b, vphi_b, rho_face, p_face):
        rho[idx] = rho_face
        mom_r[idx] = rho_face * vr_b
        mom_z[idx] = rho_face * vz_b
        mom_phi[idx] = rho_face * vphi_b
        p[idx] = p_face
        e_total[idx] = 0.5 * rho[idx] * (vr_b**2 + vz_b**2 + vphi_b**2) + p[idx]/(gamma-1.0)

    # Left/right
    set_face((0, slice(None)), +abs(v_in), 0.0, vphi_b, rho_in_profile[0, :], p_in_profile[0, :])
    set_face((-1, slice(None)), -abs(v_in), 0.0, vphi_b, rho_in_profile[-1, :], p_in_profile[-1, :])
    # Bottom/top
    set_face((slice(None), 0), 0.0, +abs(v_in), vphi_b, rho_in_profile[:, 0], p_in_profile[:, 0])
    set_face((slice(None), -1), 0.0, -abs(v_in), vphi_b, rho_in_profile[:, -1], p_in_profile[:, -1])

# ---------------- BH absorber ----------------
def apply_black_hole_absorber(rho, mom_r, mom_z, mom_phi, p, e_total, R_grid, Z_grid, Rs: float = 2.0):
    width = 3.0
    r_eff = np.sqrt(R_grid**2 + Z_grid**2)
    taper = np.ones_like(rho)
    inner = r_eff <= Rs
    mid = (r_eff > Rs) & (r_eff < Rs + width)
    taper[inner] = 0.0
    taper[mid] = (r_eff[mid] - Rs) / width
    for arr in (rho, mom_r, mom_z, mom_phi, p, e_total):
        arr *= taper

# ---------------- Compute primitives ----------------
def compute_primitives(rho, mom_r, mom_z, mom_phi, e_total, min_internal: float = MIN_INTERNAL):
    rho_safe = np.maximum(rho, 1e-16)
    vr = mom_r / rho_safe
    vz = mom_z / rho_safe
    vphi = mom_phi / rho_safe
    kinetic = 0.5 * rho_safe * (vr**2 + vz**2 + vphi**2)
    internal = e_total - kinetic
    internal = np.maximum(internal, min_internal)
    p = (GAMMA - 1.0) * internal
    return p, vr, vz, vphi
