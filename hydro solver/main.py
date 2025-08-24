#!/usr/bin/env python3
"""
main.py — updated with robust per-cell internal-energy clamps to prevent
explosive pressure / sound-speed growth.

Key behavior changes (keeps RK2, fluxes, and physics intact):
 - No global magic-scaling to tiny factors. Instead we *cap* internal energy per cell
   using a physically-motivated maximum derived from a chosen sound-speed limit CS_SAFE.
 - We avoid using gigantic sentinel constants (1e20) for energy — NaN/Inf are replaced
   with local safe values based on kinetic + MIN_INTERNAL.
 - Boundary e_total is computed consistently from capped internal when BCs are applied.
 - We report how many cells were clamped (so you don't get spammed with repeated warnings).

Put this file into the canvas and run your usual `python main.py`.
"""

import os
import time
import numpy as np
import h5py

import init_state as ist
import conservation_normal as cnorm
import conservation_parallel as cpar

# -----------------------------
# User knobs
# -----------------------------
DEBUG_SMALL = False
OUTPUT_DIR = "hydro_h5_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_INTERVAL_STEPS = 50
CFL_FACTOR = 0.35
GAMMA = ist.GAMMA

# Safety controls
MIN_RHO = 1e-6
MIN_INTERNAL = 1e-12
MIN_DT = 1e-12
VMAX_CAP = 20.0
CONS_CLAMP_RHO_MIN = 1e-8
CONS_CLAMP_RHO_MAX = 1e8
CONS_CLAMP_MOM = 50.0

# Sound-speed cap (code units) used to compute per-cell internal-energy maximum.
# Keep this safely < 1 for Newtonian runs. Match init_state.CS_SAFE for consistency.
CS_SAFE = 0.3

# -----------------------------
# Simple minimal snapshot writer (rho, p, e_total)
# -----------------------------

def save_snapshot(step, rho, p, e_total, outdir=OUTPUT_DIR):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"hydro_{step:06d}.h5")
    try:
        with h5py.File(fname, "w") as f:
            f.create_dataset("rho", data=rho, compression="gzip", compression_opts=4)
            f.create_dataset("p", data=p, compression="gzip", compression_opts=4)
            f.create_dataset("e_total", data=e_total, compression="gzip", compression_opts=4)
        print(f"[INFO] Saved {fname}")
    except Exception as e:
        print(f"[ERROR] Failed to save snapshot {fname}: {e}")

# -----------------------------
# Primitives reconstruction (consistent)
# -----------------------------

def reconstruct_primitives(e_total, rho, mom_r, mom_z, mom_phi, gamma=GAMMA, min_internal=MIN_INTERNAL):
    rho_safe = np.maximum(rho, 1e-16)
    vr = mom_r / rho_safe
    vz = mom_z / rho_safe
    vphi = mom_phi / rho_safe
    kinetic = 0.5 * rho_safe * (vr * vr + vz * vz + vphi * vphi)
    internal = e_total - kinetic
    internal = np.maximum(internal, min_internal)
    p = (gamma - 1.0) * internal
    return p, vr, vz, vphi

# -----------------------------
# Clamping / Safety
# -----------------------------

def clamp_conserved(rho, mom_r, mom_z, mom_phi, e_total):
    # Only clip the straightforward conserved quantities here — don't use huge sentinels
    rho = np.clip(rho, CONS_CLAMP_RHO_MIN, CONS_CLAMP_RHO_MAX)
    mom_r = np.clip(mom_r, -CONS_CLAMP_MOM, CONS_CLAMP_MOM)
    mom_z = np.clip(mom_z, -CONS_CLAMP_MOM, CONS_CLAMP_MOM)
    mom_phi = np.clip(mom_phi, -CONS_CLAMP_MOM, CONS_CLAMP_MOM)

    # Replace NaN/Inf in e_total with a conservative finite value: kinetic + MIN_INTERNAL
    # We'll repair internal energy properly in enforce_safety below.
    rho_safe = np.maximum(rho, MIN_RHO)
    kin_guess = 0.5 * rho_safe * ( (mom_r / (rho_safe + 1e-20))**2 + (mom_z / (rho_safe + 1e-20))**2 + (mom_phi / (rho_safe + 1e-20))**2 )
    finite_mask = np.isfinite(e_total)
    if not np.all(finite_mask):
        e_total = e_total.copy()
        e_total[~finite_mask] = kin_guess[~finite_mask] + MIN_INTERNAL

    return rho, mom_r, mom_z, mom_phi, e_total


def enforce_safety(rho, mom_r, mom_z, mom_phi, e_total, VMAX_CAP=VMAX_CAP,
                   MIN_RHO=MIN_RHO, MIN_INTERNAL=MIN_INTERNAL, CS_cap=CS_SAFE):
    """
    1) Clamp densities
    2) Cap velocities by rescaling momenta if needed
    3) Enforce internal energy floor and a physically motivated per-cell internal maximum
       derived from CS_cap (sound-speed limit), then rebuild e_total = kinetic + internal.
    """
    rho = np.clip(rho, MIN_RHO, np.finfo(float).max)

    # velocities + velocity-cap
    rho_safe = np.maximum(rho, MIN_RHO)
    vr = mom_r / rho_safe
    vz = mom_z / rho_safe
    vphi = mom_phi / rho_safe
    vmag = np.sqrt(vr * vr + vz * vz + vphi * vphi)
    mask = vmag > VMAX_CAP
    if np.any(mask):
        scale = (VMAX_CAP / (vmag + 1e-20))[mask]
        mom_r[mask] *= scale
        mom_z[mask] *= scale
        mom_phi[mask] *= scale
        # recompute velocities after scaling
        vr = mom_r / rho_safe
        vz = mom_z / rho_safe
        vphi = mom_phi / rho_safe

    # kinetic energy per cell
    kinetic = 0.5 * rho_safe * (vr * vr + vz * vz + vphi * vphi)

    # compute internal energy and clamp to [MIN_INTERNAL, internal_max]
    internal = e_total - kinetic
    # replace non-finite internals with MIN_INTERNAL
    internal = np.where(np.isfinite(internal), internal, MIN_INTERNAL)
    internal = np.maximum(internal, MIN_INTERNAL)

    # per-cell maximum internal derived from CS_cap
    # internal_max = rho * cs^2 / (gamma*(gamma-1))  (from cs^2 = gamma*(gamma-1)*internal / rho)
    denom = (GAMMA * (GAMMA - 1.0))
    internal_max = rho_safe * (CS_cap ** 2) / denom

    # clamp and count how many cells were above the limit
    above = internal > internal_max
    n_above = int(np.count_nonzero(above))
    if n_above > 0:
        internal[above] = internal_max[above]

    e_total = kinetic + internal

    if n_above > 0:
        print(f"[WARN] enforce_safety: clamped internal energy in {n_above} cell(s) to enforce cs<={CS_cap:.3e}")

    return rho, mom_r, mom_z, mom_phi, e_total

# -----------------------------
# CFL dt
# -----------------------------

def compute_dt_cfl_safe(rho, mom_r, mom_z, mom_phi, e_total, dr, dz, prev_dt=None):
    rho_safe = np.maximum(rho, MIN_RHO)
    p_local, vr, vz, vphi = reconstruct_primitives(e_total, rho, mom_r, mom_z, mom_phi)
    cs = np.sqrt(np.maximum(GAMMA * p_local / rho_safe, 0.0))
    vmag = np.sqrt(vr * vr + vz * vz + vphi * vphi)
    vmag = np.minimum(vmag, VMAX_CAP)
    s = vmag + cs
    s = np.nan_to_num(s, nan=0.0, posinf=VMAX_CAP, neginf=0.0)
    vmax = float(np.max(s))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return max(MIN_DT, 1e-6)
    raw_dt = CFL_FACTOR * min(dr, dz) / (vmax + 1e-12)
    dt = max(raw_dt, MIN_DT)
    if prev_dt is not None and prev_dt > 0:
        dt = min(dt, prev_dt * 5.0)
    return float(dt)

# -----------------------------
# Apply conserved-variable BC helper (ensures e_total consistent with capped internal)
# -----------------------------

def apply_conserved_bcs(rho, mom_r, mom_z, mom_phi, e_total, R_grid, Z_grid, params, CS_cap=CS_SAFE):
    # prefer user-supplied apply_boundaries if it implements conserved BCs
    if hasattr(ist, 'apply_boundaries'):
        try:
            # call with signature used by init_state.apply_boundaries
            ist.apply_boundaries(rho, mom_r, mom_z, mom_phi, None, e_total, R_grid, Z_grid, params)
            return
        except TypeError:
            pass

    # fallback: set e_total from primitives in params (if provided)
    p_in = params.get('p_in', params.get('p_inj', None))
    rho_in = params.get('rho_in', params.get('rho_inj', None))
    v_in = params.get('v_in', 0.0)
    vphi_b = params.get('vphi_b', 0.0)
    if p_in is None or rho_in is None:
        return

    # compute internal from p and clamp to internal_max
    rho_idx = rho_in
    p_idx = p_in
    # compute internal and clamp per cell
    internal_idx = p_idx / (GAMMA - 1.0)
    denom = (GAMMA * (GAMMA - 1.0))
    internal_max_idx = rho_idx * (CS_cap ** 2) / denom
    internal_idx = np.minimum(internal_idx, internal_max_idx)

    # left/right
    for idx, vr_val in [((0, slice(None)), +abs(v_in)), ((-1, slice(None)), -abs(v_in))]:
        rho[idx] = rho_idx[idx]
        mom_r[idx] = rho_idx[idx] * vr_val
        mom_z[idx] = 0.0
        mom_phi[idx] = rho_idx[idx] * vphi_b
        e_total[idx] = 0.5 * rho[idx] * (vr_val**2) + internal_idx[idx]

    # bottom/top
    for idx, vz_val in [((slice(None), 0), +abs(v_in)), ((slice(None), -1), -abs(v_in))]:
        rho[idx] = rho_idx[idx]
        mom_r[idx] = 0.0
        mom_z[idx] = rho_idx[idx] * vz_val
        mom_phi[idx] = rho_idx[idx] * vphi_b
        e_total[idx] = 0.5 * rho[idx] * (vz_val**2) + internal_idx[idx]

# -----------------------------
# Main loop
# -----------------------------

def main():
    # grid
    if DEBUG_SMALL:
        r, z, dr, dz, R_grid, Z_grid = ist.make_grid(nr=128, nz=128, Rmax=50.0, Zmax=50.0)
    else:
        r, z, dr, dz, R_grid, Z_grid = ist.make_grid()

    cnorm.set_spacings(dr, dz, gamma=GAMMA)
    try:
        cpar.set_spacings(dr, dz)
    except Exception:
        pass

    rho, p, vr, vz, vphi, mom_r, mom_z, mom_phi, e_total, params = ist.initial_conditions(R_grid, Z_grid)

    # potential arrays
    phi_arr = cnorm.np_paczynski_wiita_potential(R_grid, Z_grid)
    try:
        dPhi_dr_arr, dPhi_dz_arr = cpar.grad_potential_pw(R_grid, Z_grid)
    except Exception:
        dPhi_dr_arr, dPhi_dz_arr = cnorm.np_grad_potential_pw(R_grid, Z_grid)

    # ensure types
    rho = rho.astype(np.float64)
    mom_r = mom_r.astype(np.float64)
    mom_z = mom_z.astype(np.float64)
    mom_phi = mom_phi.astype(np.float64)
    e_total = e_total.astype(np.float64)

    # initial clamp & safety
    rho, mom_r, mom_z, mom_phi, e_total = clamp_conserved(rho, mom_r, mom_z, mom_phi, e_total)
    rho, mom_r, mom_z, mom_phi, e_total = enforce_safety(rho, mom_r, mom_z, mom_phi, e_total)

    # initial primitives
    p, vr, vz, vphi = reconstruct_primitives(e_total, rho, mom_r, mom_z, mom_phi)

    t = 0.0
    step = 0
    prev_dt = None

    max_steps = 10000 if not DEBUG_SMALL else 200

    save_snapshot(step, rho, p, e_total)

    try:
        while step < max_steps:
            # clamp conserved
            rho, mom_r, mom_z, mom_phi, e_total = clamp_conserved(rho, mom_r, mom_z, mom_phi, e_total)
            rho, mom_r, mom_z, mom_phi, e_total = enforce_safety(rho, mom_r, mom_z, mom_phi, e_total)

            # primitives
            p_safe, vr, vz, vphi = reconstruct_primitives(e_total, rho, mom_r, mom_z, mom_phi)

            # timestep
            dt = compute_dt_cfl_safe(rho, mom_r, mom_z, mom_phi, e_total, dr, dz, prev_dt)
            prev_dt = dt
            if t + dt > 1e6:
                break

            # RHS
            drho_dt = cnorm.np_continuity_eqn(rho, vr, vz, R_grid)
            dEr_dt = cnorm.np_momentum_r(rho, vr, vz, p_safe, vphi, dPhi_dr_arr, R_grid)
            dEz_dt = cnorm.np_momentum_z(rho, vr, vz, p_safe, dPhi_dz_arr, R_grid)
            dEphi_dt = cnorm.np_angular_mom(mom_r, mom_z, mom_phi, rho, R_grid)
            dEtot_dt = cnorm.np_energy(rho, vr, vz, vphi, p_safe, e_total, dPhi_dr_arr, dPhi_dz_arr, R_grid)

            # predictor
            rho1 = rho + dt * drho_dt
            mom_r1 = mom_r + dt * dEr_dt
            mom_z1 = mom_z + dt * dEz_dt
            mom_phi1 = mom_phi + dt * dEphi_dt
            e_total1 = e_total + dt * dEtot_dt

            # clamp predictor
            rho1, mom_r1, mom_z1, mom_phi1, e_total1 = clamp_conserved(rho1, mom_r1, mom_z1, mom_phi1, e_total1)
            rho1, mom_r1, mom_z1, mom_phi1, e_total1 = enforce_safety(rho1, mom_r1, mom_z1, mom_phi1, e_total1)

            # reconstruct and apply BCs
            p1, vr1, vz1, vphi1 = reconstruct_primitives(e_total1, rho1, mom_r1, mom_z1, mom_phi1)
            apply_conserved_bcs(rho1, mom_r1, mom_z1, mom_phi1, e_total1, R_grid, Z_grid, params, CS_cap=CS_SAFE)

            # RHS at predictor
            drho_dt2 = cnorm.np_continuity_eqn(rho1, vr1, vz1, R_grid)
            dEr_dt2 = cnorm.np_momentum_r(rho1, vr1, vz1, p1, vphi1, dPhi_dr_arr, R_grid)
            dEz_dt2 = cnorm.np_momentum_z(rho1, vr1, vz1, p1, dPhi_dz_arr, R_grid)
            dEphi_dt2 = cnorm.np_angular_mom(mom_r1, mom_z1, mom_phi1, rho1, R_grid)
            dEtot_dt2 = cnorm.np_energy(rho1, vr1, vz1, vphi1, p1, e_total1, dPhi_dr_arr, dPhi_dz_arr, R_grid)

            # corrector
            rho = 0.5 * (rho + rho1 + dt * drho_dt2)
            mom_r = 0.5 * (mom_r + mom_r1 + dt * dEr_dt2)
            mom_z = 0.5 * (mom_z + mom_z1 + dt * dEz_dt2)
            mom_phi = 0.5 * (mom_phi + mom_phi1 + dt * dEphi_dt2)
            e_total = 0.5 * (e_total + e_total1 + dt * dEtot_dt2)

            # final clamps + BCs
            rho, mom_r, mom_z, mom_phi, e_total = clamp_conserved(rho, mom_r, mom_z, mom_phi, e_total)
            rho, mom_r, mom_z, mom_phi, e_total = enforce_safety(rho, mom_r, mom_z, mom_phi, e_total)

            apply_conserved_bcs(rho, mom_r, mom_z, mom_phi, e_total, R_grid, Z_grid, params, CS_cap=CS_SAFE)

            # final primitives
            p, vr, vz, vphi = reconstruct_primitives(e_total, rho, mom_r, mom_z, mom_phi)

            # diagnostics
            if step % 10 == 0:
                csmax = np.sqrt(np.maximum(GAMMA * p / np.maximum(rho, 1e-16), 0.0)).max()
                print(f"step={step} t={t:.3e} dt={dt:.3e} rho[max]={rho.max():.3e} p[max]={p.max():.3e} csmax={csmax:.3e}")

            if step % OUTPUT_INTERVAL_STEPS == 0:
                save_snapshot(step, rho, p, e_total)

            t += dt
            step += 1

    except KeyboardInterrupt:
        print('[INFO] Interrupted by user.')
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in main loop: {e}")
        traceback.print_exc()
    finally:
        print('Done. final step:', step)


if __name__ == '__main__':
    main()
