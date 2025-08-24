#!/usr/bin/env python3
# conservation_normal.py — nonrelativistic-safe, cylindrical (r–z)
"""
Vector-free (NumPy) implementation of the Euler equations in axisymmetric
cylindrical coordinates (r, z) with a fixed gravitational potential.

This module supplies numerically stable, nonrelativistic fluxes that are
consistent with the solver in `main.py` and the initial/boundary logic in
`init_state.py`.  It uses TVD minmod reconstruction and a local
Lax–Friedrichs (Rusanov) numerical flux.  Geometry terms are handled via the
"1/r * d(r F_r)/dr" form to keep the radial pressure term conservative.

Conserved variables advanced in `main.py`:
  ρ                (density)
  ρ v_r            (radial momentum)
  ρ v_z            (vertical momentum)
  ρ v_φ            (azimuthal momentum; evolved in non-conservative form here)
  E = u + ½ρ|v|²   (total energy excluding gravitational potential energy)

Equations (with external potential Φ):
  ∂t ρ + 1/r ∂r[r ρ v_r] + ∂z[ρ v_z] = 0
  ∂t(ρ v_r) + 1/r ∂r[r (ρ v_r² + p)] + ∂z[ρ v_r v_z] − ρ v_φ² / r = −ρ ∂rΦ
  ∂t(ρ v_z) + 1/r ∂r[r (ρ v_r v_z)] + ∂z[ρ v_z² + p] = −ρ ∂zΦ
  ∂t E + 1/r ∂r[r (E + p) v_r] + ∂z[(E + p) v_z] = −ρ (v_r ∂rΦ + v_z ∂zΦ)
  ∂t(ρ v_φ) + 1/r ∂r[r ρ v_r v_φ] + ∂z[ρ v_z v_φ] + (ρ v_r v_φ)/r = 0

All routines return the *RHS* for an explicit update, i.e. −∇·F + S, so that
`U^{n+1} = U^n + dt * RHS`.

Exported API:
  set_spacings(dr, dz, gamma=5/3)
  np_paczynski_wiita_potential(R, Z, GM=1.0, Rs=2.0)
  np_grad_potential_pw(R, Z, GM=1.0, Rs=2.0)
  np_continuity_eqn(rho, vr, vz, R_grid)
  np_momentum_r(rho, vr, vz, p, vphi, dPhi_dr, R_grid)
  np_momentum_z(rho, vr, vz, p, dPhi_dz)
  np_angular_mom(mom_r, mom_z, mom_phi, rho, R_grid)
  np_energy(rho, vr, vz, vphi, p, e_tot, dPhi_dr, dPhi_dz, R_grid)
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# Globals set by set_spacings
# ----------------------------------------------------------------------------
dr: float = 1.0
dz: float = 1.0
gamma: float = 5.0/3.0

# Small numerical guards
_EPS_R = 1e-9
_EPS = 1e-30

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

def set_spacings(dr_in: float, dz_in: float, gamma: float = 5.0/3.0):
    """Configure grid spacings and adiabatic index used by flux routines."""
    globals()['dr'] = float(dr_in)
    globals()['dz'] = float(dz_in)
    globals()['gamma'] = float(gamma)

# ----------------------------------------------------------------------------
# Utilities: reconstruction + Rusanov flux
# ----------------------------------------------------------------------------

def _minmod(a, b):
    s = np.sign(a) + np.sign(b)
    return 0.5 * s * np.minimum(np.abs(a), np.abs(b))


def _reconstruct_lr(q: np.ndarray, axis: int):
    """TVD minmod piecewise-linear reconstruction.
    Returns left/right interface states q_L, q_R at i+1/2 faces along `axis`.
    Shapes match the cell-centered q except missing one cell along `axis`.
    """
    dq = np.diff(q, axis=axis)
    # pad for slope left/right
    dq_m = np.take(dq, indices=range(dq.shape[axis]-1), axis=axis)
    dq_p = np.take(dq, indices=range(1, dq.shape[axis]), axis=axis)
    slope = _minmod(dq_m, dq_p)

    # build q_L and q_R at faces i+1/2
    # q_R at i-1/2 equals q_L at i-1/2 of the next cell; construct explicitly
    # Center values excluding first/last along axis
    q_c = np.take(q, indices=range(1, q.shape[axis]-1), axis=axis)
    q_L = q_c + 0.5 * slope
    q_R = q_c - 0.5 * slope
    return q_L, q_R


def _rusanov_flux(q_L, q_R, u_L, u_R):
    """Local Lax–Friedrichs (Rusanov) numerical flux for scalar advection F = q * u.
    Returns flux defined at faces.
    """
    a = np.maximum(np.abs(u_L), np.abs(u_R))
    return 0.5 * (q_L * u_L + q_R * u_R) - 0.5 * a * (q_R - q_L)


def _face_average(a: np.ndarray, axis: int):
    """Simple arithmetic average to faces along `axis` (i+1/2)."""
    return 0.5 * (np.take(a, range(0, a.shape[axis]-2), axis=axis) +
                  np.take(a, range(2, a.shape[axis]), axis=axis))

# ----------------------------------------------------------------------------
# Potential: Paczyński–Wiita (pseudo-Newtonian)
# ----------------------------------------------------------------------------

def np_paczynski_wiita_potential(R: np.ndarray, Z: np.ndarray, GM: float = 1.0, Rs: float = 2.0):
    """Φ = − GM / (sqrt(R^2 + Z^2) − Rs).  Outside r>Rs only; internally we cap."""
    s = np.sqrt(R*R + Z*Z)
    denom = np.maximum(s - Rs, 1e-8)
    return -GM / denom


def np_grad_potential_pw(R: np.ndarray, Z: np.ndarray, GM: float = 1.0, Rs: float = 2.0):
    """Gradient of the Paczyński–Wiita potential (finite outside r>Rs).
    dΦ/ds = + GM / (s − Rs)^2,  ∇s = (R/s, Z/s).
    """
    s = np.sqrt(R*R + Z*Z)
    s_safe = np.maximum(s, 1e-8)
    denom = np.maximum(s - Rs, 1e-8)
    pref = GM / (denom*denom * s_safe)
    return pref * R, pref * Z

# ----------------------------------------------------------------------------
# Flux helpers: divergence operators (return −div F with proper geometry)
# ----------------------------------------------------------------------------

def _divergence_r_geom(F_face_r: np.ndarray, R_grid: np.ndarray, dr):
    """Geometric divergence in r: − (1/r) ∂r[ r F_r ] using face-centered F_r.
    F_face_r is defined at i+1/2 faces (shape (nr+1, nz)), R_grid is cell-centered (nr, nz).
    """
    nr, nz = R_grid.shape
    dr_local = dr if dr is not None else globals().get('dr', 1.0)
    # Build radial faces (nr+1)
    r_faces = np.linspace(R_grid[:,0].min() - dr_local/2,
                          R_grid[:,0].max() + dr_local/2,
                          nr+1)  # length = nr+1

    # Compute r*F at faces
    RF = r_faces[:, None] * F_face_r  # shape (nr+1, nz)

    # Divergence inside cells (nr, nz)
    div_r = (RF[1:, :] - RF[:-1, :]) / (dr_local * R_grid)
    return div_r


def _divergence_z(F_face_z: np.ndarray):
    """Cartesian divergence in z: − ∂z F_z using face-centered F_z."""
    num = F_face_z[:, 1:] - F_face_z[:, :-1]
    out = - num / dz
    # pad to full shape
    left = out[:, 0:1]
    right = out[:, -1:]
    out_full = np.hstack([left, out, right])
    return out_full

# ----------------------------------------------------------------------------
# Public RHS builders
# ----------------------------------------------------------------------------

def np_continuity_eqn(rho: np.ndarray, vr: np.ndarray, vz: np.ndarray, R_grid: np.ndarray):
    """RHS for continuity: −[ 1/r ∂r(r ρ v_r) + ∂z(ρ v_z) ]."""
    # fluxes must be face-centered
    nr, nz = rho.shape
    dr_local = dr
    dz_local = dz
    # Build face-centered arrays (nr+1, nz)
    rho_faces = np.zeros((nr+1, nz), dtype=rho.dtype)
    vr_faces = np.zeros((nr+1, nz), dtype=vr.dtype)
    # Internal faces: average adjacent cells
    rho_faces[1:-1, :] = 0.5 * (rho[1:, :] + rho[:-1, :])
    vr_faces[1:-1, :] = 0.5 * (vr[1:, :] + vr[:-1, :])
    # Boundaries: copy edge values
    rho_faces[0, :] = rho[0, :]
    rho_faces[-1, :] = rho[-1, :]
    vr_faces[0, :] = vr[0, :]
    vr_faces[-1, :] = vr[-1, :]
    F_r = rho_faces * vr_faces  # shape (nr+1, nz)

    # z fluxes (nr, nz+1)
    rho_zfaces = np.zeros((nr, nz+1), dtype=rho.dtype)
    vz_zfaces = np.zeros((nr, nz+1), dtype=vz.dtype)
    rho_zfaces[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
    vz_zfaces[:, 1:-1] = 0.5 * (vz[:, 1:] + vz[:, :-1])
    rho_zfaces[:, 0] = rho[:, 0]
    rho_zfaces[:, -1] = rho[:, -1]
    vz_zfaces[:, 0] = vz[:, 0]
    vz_zfaces[:, -1] = vz[:, -1]
    F_z = rho_zfaces * vz_zfaces  # shape (nr, nz+1)

    div_r = _divergence_r_geom(F_r, R_grid, dr_local)      # (nr, nz)
    div_z = (F_z[:, 1:] - F_z[:, :-1]) / dz_local          # (nr, nz)

    return div_r + div_z


def np_momentum_r(rho: np.ndarray, vr: np.ndarray, vz: np.ndarray, p: np.ndarray,
                  vphi: np.ndarray, dPhi_dr: np.ndarray, R_grid: np.ndarray):
    """RHS for radial momentum:
    −[ 1/r ∂r(r (ρ v_r² + p)) + ∂z(ρ v_r v_z) ]  +  (ρ v_φ²)/r  −  ρ ∂rΦ
    """
    nr, nz = rho.shape
    dr_local = dr
    dz_local = dz
    # Build face-centered arrays (nr+1, nz) for (ρ v_r^2 + p)
    rho_faces = np.zeros((nr+1, nz), dtype=rho.dtype)
    vr_faces = np.zeros((nr+1, nz), dtype=vr.dtype)
    p_faces = np.zeros((nr+1, nz), dtype=p.dtype)
    rho_faces[1:-1, :] = 0.5 * (rho[1:, :] + rho[:-1, :])
    vr_faces[1:-1, :] = 0.5 * (vr[1:, :] + vr[:-1, :])
    p_faces[1:-1, :] = 0.5 * (p[1:, :] + p[:-1, :])
    rho_faces[0, :] = rho[0, :]
    rho_faces[-1, :] = rho[-1, :]
    vr_faces[0, :] = vr[0, :]
    vr_faces[-1, :] = vr[-1, :]
    p_faces[0, :] = p[0, :]
    p_faces[-1, :] = p[-1, :]
    F_r = rho_faces * vr_faces * vr_faces + p_faces  # shape (nr+1, nz)

    # z fluxes (nr, nz+1) for (ρ v_r v_z)
    rho_zfaces = np.zeros((nr, nz+1), dtype=rho.dtype)
    vr_zfaces = np.zeros((nr, nz+1), dtype=vr.dtype)
    vz_zfaces = np.zeros((nr, nz+1), dtype=vz.dtype)
    rho_zfaces[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
    vr_zfaces[:, 1:-1] = 0.5 * (vr[:, 1:] + vr[:, :-1])
    vz_zfaces[:, 1:-1] = 0.5 * (vz[:, 1:] + vz[:, :-1])
    rho_zfaces[:, 0] = rho[:, 0]
    rho_zfaces[:, -1] = rho[:, -1]
    vr_zfaces[:, 0] = vr[:, 0]
    vr_zfaces[:, -1] = vr[:, -1]
    vz_zfaces[:, 0] = vz[:, 0]
    vz_zfaces[:, -1] = vz[:, -1]
    F_z = rho_zfaces * vr_zfaces * vz_zfaces  # shape (nr, nz+1)

    geom_src = (rho * vphi * vphi) / np.maximum(R_grid, _EPS_R)
    grav_src = - rho * dPhi_dr

    return _divergence_r_geom(F_r, R_grid, dr_local) + (F_z[:, 1:] - F_z[:, :-1]) / dz_local + geom_src + grav_src


def np_momentum_z(rho: np.ndarray, vr: np.ndarray, vz: np.ndarray, p: np.ndarray,
                  dPhi_dz: np.ndarray, R_grid: np.ndarray, dr=1.0, dz=1.0):
    """RHS for vertical momentum:
    −[ 1/r ∂r(r ρ v_r v_z) + ∂z(ρ v_z² + p) ] − ρ ∂zΦ
    """
    nr, nz = rho.shape
    dr_local = dr
    dz_local = dz
    # Build face-centered arrays (nr+1, nz) for (ρ v_r v_z)
    rho_faces = np.zeros((nr+1, nz), dtype=rho.dtype)
    vr_faces = np.zeros((nr+1, nz), dtype=vr.dtype)
    vz_faces = np.zeros((nr+1, nz), dtype=vz.dtype)
    rho_faces[1:-1, :] = 0.5 * (rho[1:, :] + rho[:-1, :])
    vr_faces[1:-1, :] = 0.5 * (vr[1:, :] + vr[:-1, :])
    vz_faces[1:-1, :] = 0.5 * (vz[1:, :] + vz[:-1, :])
    rho_faces[0, :] = rho[0, :]
    rho_faces[-1, :] = rho[-1, :]
    vr_faces[0, :] = vr[0, :]
    vr_faces[-1, :] = vr[-1, :]
    vz_faces[0, :] = vz[0, :]
    vz_faces[-1, :] = vz[-1, :]
    F_r = rho_faces * vr_faces * vz_faces  # shape (nr+1, nz)

    # z fluxes (nr, nz+1) for (ρ v_z^2 + p)
    rho_zfaces = np.zeros((nr, nz+1), dtype=rho.dtype)
    vz_zfaces = np.zeros((nr, nz+1), dtype=vz.dtype)
    p_zfaces = np.zeros((nr, nz+1), dtype=p.dtype)
    rho_zfaces[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
    vz_zfaces[:, 1:-1] = 0.5 * (vz[:, 1:] + vz[:, :-1])
    p_zfaces[:, 1:-1] = 0.5 * (p[:, 1:] + p[:, :-1])
    rho_zfaces[:, 0] = rho[:, 0]
    rho_zfaces[:, -1] = rho[:, -1]
    vz_zfaces[:, 0] = vz[:, 0]
    vz_zfaces[:, -1] = vz[:, -1]
    p_zfaces[:, 0] = p[:, 0]
    p_zfaces[:, -1] = p[:, -1]
    F_z = rho_zfaces * vz_zfaces * vz_zfaces + p_zfaces  # shape (nr, nz+1)

    grav_src = - rho * dPhi_dz

    return _divergence_r_geom(F_r, R_grid, dr_local) + (F_z[:, 1:] - F_z[:, :-1]) / dz_local + grav_src



def np_angular_mom(mom_r: np.ndarray, mom_z: np.ndarray, mom_phi: np.ndarray,
                   rho: np.ndarray, R_grid: np.ndarray):
    """RHS for azimuthal momentum (ρ v_φ) in non-conservative form:
    −[ 1/r ∂r(r ρ v_r v_φ) + ∂z(ρ v_z v_φ) + (ρ v_r v_φ)/r ].
    This avoids the explicit evolution of L = ρ r v_φ while remaining stable.
    """
    vr = mom_r / np.maximum(rho, _EPS)
    vz = mom_z / np.maximum(rho, _EPS)
    vphi = mom_phi / np.maximum(rho, _EPS)

    # r flux of ρ v_r v_φ
    q = rho * vr * vphi
    qL_r, qR_r = _reconstruct_lr(q, axis=0)
    uL_r, uR_r = _reconstruct_lr(vr, axis=0)
    F_r = _rusanov_flux(qL_r, qR_r, uL_r, uR_r)

    # z flux of ρ v_z v_φ
    q = rho * vz * vphi
    qL_z, qR_z = _reconstruct_lr(q, axis=1)
    uL_z, uR_z = _reconstruct_lr(vz, axis=1)
    F_z = _rusanov_flux(qL_z, qR_z, uL_z, uR_z)

    geom_src = - (rho * vr * vphi) / np.maximum(R_grid, _EPS_R)

    dr_local = 1.0  # TODO: pass actual dr if needed
    return _divergence_r_geom(F_r, R_grid, dr_local) + _divergence_z(F_z) + geom_src


def np_energy(rho: np.ndarray, vr: np.ndarray, vz: np.ndarray, vphi: np.ndarray,
              p: np.ndarray, e_tot: np.ndarray,
              dPhi_dr: np.ndarray, dPhi_dz: np.ndarray, R_grid: np.ndarray):
    """RHS for energy (E = internal + kinetic), excluding gravitational energy:
    −[ 1/r ∂r(r (E + p) v_r) + ∂z((E + p) v_z) ] − ρ (v_r ∂rΦ + v_z ∂zΦ)
    """
    H = e_tot + p

    # r flux of (E+p) v_r
    q = H
    u = vr
    qL_r, qR_r = _reconstruct_lr(q, axis=0)
    uL_r, uR_r = _reconstruct_lr(u, axis=0)
    F_r = _rusanov_flux(qL_r, qR_r, uL_r, uR_r)

    # z flux of (E+p) v_z
    qL_z, qR_z = _reconstruct_lr(q, axis=1)
    uL_z, uR_z = _reconstruct_lr(vz, axis=1)
    F_z = _rusanov_flux(qL_z, qR_z, uL_z, uR_z)

    grav_src = - rho * (vr * dPhi_dr + vz * dPhi_dz)

    dr_local = 1.0  # TODO: pass actual dr if needed
    return _divergence_r_geom(F_r, R_grid, dr_local) + _divergence_z(F_z) + grav_src
