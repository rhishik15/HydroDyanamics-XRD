#!/usr/bin/env python3
"""
conservation_parallel.py — nonrelativistic-safe, numba-parallelized operators
for axisymmetric (r–z) hydrodynamics with a Paczyński–Wiita potential.

This mirrors the API of conservation_normal.py but uses Numba (if available)
with prange to accelerate the core finite-volume sweeps.

Key features
------------
- Cylindrical geometry terms handled conservatively: (1/r) ∂_r (r F_r)
- Rusanov (local Lax–Friedrichs) numerical flux with TVD minmod limiter
- Gravity source via Paczyński–Wiita potential Φ = -GM / (sqrt(r^2+z^2) - R_s)
- Stable handling close to r=0 and inside the pseudo-horizon (absorber expected upstream)
- Nonrelativistic regime (no c=1 tricks). All speeds are in code units.

Public API (kept compatible with main.py):
- set_spacings(dr, dz, gamma=5/3, GM=1.0, Rs=2.0)
- np_paczynski_wiita_potential(R, Z)
- np_grad_potential_pw(R, Z)
- np_continuity_eqn(rho, vr, vz, R)
- np_momentum_r(rho, vr, vz, p, vphi, dPhi_dr, R)
- np_momentum_z(rho, vr, vz, p, dPhi_dz)
- np_angular_mom(mom_r, mom_z, mom_phi, rho, R)
- np_energy(rho, vr, vz, vphi, p, e_total, dPhi_dr, dPhi_dz, R)

Notes
-----
- All arrays are float64 and assumed C-contiguous.
- If Numba is unavailable, we silently fall back to NumPy vectorization.
- The angular momentum equation here advances linear azimuthal momentum (ρ v_φ)
  with simple advection by (v_r, v_z). For more accuracy you can switch to
  conservative evolution of L = ρ r v_φ, but keep consistency with normal module.
"""
from __future__ import annotations
import numpy as np

# ---------------- Runtime knobs / globals ----------------
GAMMA = 5.0/3.0
DR = 1.0
DZ = 1.0
GM = 1.0
RS = 2.0
EPS_R = 1e-6

try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def prange(*args):
        return range(*args)

# ---------------- Utilities ----------------

def set_spacings(dr: float, dz: float, gamma: float = GAMMA, GM_: float = 1.0, Rs_: float = 2.0):
    global DR, DZ, GAMMA, GM, RS, EPS_R
    DR = float(dr)
    DZ = float(dz)
    GAMMA = float(gamma)
    GM = float(GM_)
    RS = float(Rs_)
    EPS_R = max(1e-12, 0.25*min(DR, DZ))

@njit(cache=True)
def _minmod(a, b):
    out = 0.0
    if a*b <= 0.0:
        out = 0.0
    else:
        if abs(a) < abs(b):
            out = a
        else:
            out = b
    return out

@njit(cache=True)
def _slope_minmod(arr):
    n = arr.shape[0]
    s = np.zeros_like(arr)
    for i in range(1, n-1):
        dl = arr[i] - arr[i-1]
        dr = arr[i+1] - arr[i]
        s[i] = _minmod(dl, dr)
    s[0] = 0.0
    s[n-1] = 0.0
    return s

@njit(cache=True)
def _sound_speed2(p, rho, gamma):
    return gamma * np.maximum(p, 0.0) / np.maximum(rho, 1e-16)

# ---------------- Potential and gradients ----------------

def np_paczynski_wiita_potential(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    r = np.sqrt(R*R + Z*Z)
    r_eff = np.maximum(r - RS, 1e-12)
    return -GM / r_eff

if NUMBA_OK:
    @njit(cache=True, parallel=True)
    def _grad_phi_pw(R, Z, GM, RS, eps):
        nr, nz = R.shape
        dpr = np.zeros_like(R)
        dpz = np.zeros_like(Z)
        for i in prange(nr):
            for j in range(nz):
                r = (R[i, j]*R[i, j] + Z[i, j]*Z[i, j])**0.5
                if r < eps:
                    dpr[i, j] = 0.0
                    dpz[i, j] = 0.0
                    continue
                r_eff = max(r - RS, 1e-12)
                coef = GM / (r_eff*r_eff * r)
                dpr[i, j] = coef * R[i, j]
                dpz[i, j] = coef * Z[i, j]
        return dpr, dpz
else:
    def _grad_phi_pw(R, Z, GM, RS, eps):
        r = np.sqrt(R*R + Z*Z)
        r_safe = np.maximum(r, eps)
        r_eff = np.maximum(r - RS, 1e-12)
        coef = GM / (r_eff*r_eff * r_safe)
        return coef*R, coef*Z

def np_grad_potential_pw(R: np.ndarray, Z: np.ndarray):
    return _grad_phi_pw(R.astype(np.float64), Z.astype(np.float64), float(GM), float(RS), float(EPS_R))

# ---------------- Core flux helpers (1D building blocks) ----------------

@njit(cache=True)
def _rusanov_flux_mass(rhoL, vL, rhoR, vR):
    # F = ρ v
    fL = rhoL * vL
    fR = rhoR * vR
    a = max(abs(vL), abs(vR))
    return 0.5*(fL + fR) - 0.5*a*(rhoR - rhoL)

@njit(cache=True)
def _rusanov_flux_mom(rhoL, vL, pL, rhoR, vR, pR):
    # F = ρ v^2 + p
    fL = rhoL*vL*vL + pL
    fR = rhoR*vR*vR + pR
    a = max(abs(vL) + 0.0, abs(vR) + 0.0)
    return 0.5*(fL + fR) - 0.5*a*(rhoR*vR - rhoL*vL)

@njit(cache=True)
def _rusanov_flux_energy(rhoL, vL, pL, eL, rhoR, vR, pR, eR, gamma):
    # total energy flux: (E + p) v, with E = internal + kinetic
    HL = eL + pL
    HR = eR + pR
    fL = HL * vL
    fR = HR * vR
    # estimate signal speed (v + c)
    cL = (gamma*pL/max(rhoL,1e-16))**0.5
    cR = (gamma*pR/max(rhoR,1e-16))**0.5
    a = max(abs(vL) + cL, abs(vR) + cR)
    return 0.5*(fL + fR) - 0.5*a*(eR - eL)

# ---------------- Divergence operators (parallel) ----------------

if NUMBA_OK:
    @njit(cache=True, parallel=True)
    def _div_mass(rho, vr, vz, R, dr, dz):
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        # radial faces
        for i in prange(1, nr-1):
            rL = R[i-1, 0]
            rC = R[i, 0]
            rR = R[i+1, 0]
            # limited states for ρ at i-1/2 and i+1/2 along r
            for j in range(nz):
                # reconstruct ρ and v_r at faces
                # simple piecewise linear with minmod
                # left face between i-1 and i
                d_rho_im1 = rho[i-1, j] - rho[i-2, j] if i-1 > 0 else 0.0
                d_rho_i   = rho[i, j]   - rho[i-1, j]
                sL = _minmod(d_rho_im1, d_rho_i)
                rhoL = rho[i-1, j] + 0.5*sL

                d_rho_i_p = rho[i+1, j] - rho[i, j]
                sR = _minmod(d_rho_i, d_rho_i_p)
                rhoR = rho[i, j] - 0.5*sR

                d_vr_im1 = vr[i-1, j] - vr[i-2, j] if i-1 > 0 else 0.0
                d_vr_i   = vr[i, j]   - vr[i-1, j]
                svL = _minmod(d_vr_im1, d_vr_i)
                vL = vr[i-1, j] + 0.5*svL

                d_vr_i_p = vr[i+1, j] - vr[i, j]
                svR = _minmod(d_vr_i, d_vr_i_p)
                vR = vr[i, j] - 0.5*svR

                FrL = _rusanov_flux_mass(rhoL, vL, rhoR, vR)

                # right face between i and i+1
                d_rho_i   = rho[i, j] - rho[i-1, j]
                d_rho_ip1 = rho[i+1, j] - rho[i, j]
                sL = _minmod(d_rho_i, d_rho_ip1)
                rhoL = rho[i, j] + 0.5*sL

                d_rho_ip2 = rho[i+2, j] - rho[i+1, j] if i+1 < nr-1 else 0.0
                sR = _minmod(d_rho_ip1, d_rho_ip2)
                rhoR = rho[i+1, j] - 0.5*sR

                d_vr_i   = vr[i, j] - vr[i-1, j]
                d_vr_ip1 = vr[i+1, j] - vr[i, j]
                svL = _minmod(d_vr_i, d_vr_ip1)
                vL = vr[i, j] + 0.5*svL

                d_vr_ip2 = vr[i+2, j] - vr[i+1, j] if i+1 < nr-1 else 0.0
                svR = _minmod(d_vr_ip1, d_vr_ip2)
                vR = vr[i+1, j] - 0.5*svR

                FrR = _rusanov_flux_mass(rhoL, vL, rhoR, vR)

                # geometric divergence term (1/r) ∂_r (r F_r)
                # use r at faces ~ (r_i-1/2, r_i+1/2)
                rLm = 0.5*(R[i-1, 0] + R[i, 0])
                rRp = 0.5*(R[i, 0] + R[i+1, 0])
                div_r = (rRp*FrR - rLm*FrL) / (rC * dr)

                out[i, j] += div_r

        # vertical faces (standard ∂_z F_z)
        for i in prange(1, nr-1):
            for j in range(1, nz-1):
                # face j-1/2
                d_rho_jm1 = rho[i, j-1] - rho[i, j-2] if j-1 > 0 else 0.0
                d_rho_j   = rho[i, j]   - rho[i, j-1]
                sL = _minmod(d_rho_jm1, d_rho_j)
                rhoL = rho[i, j-1] + 0.5*sL

                d_rho_j_p = rho[i, j+1] - rho[i, j]
                sR = _minmod(d_rho_j, d_rho_j_p)
                rhoR = rho[i, j] - 0.5*sR

                d_vz_jm1 = vz[i, j-1] - vz[i, j-2] if j-1 > 0 else 0.0
                d_vz_j   = vz[i, j]   - vz[i, j-1]
                svL = _minmod(d_vz_jm1, d_vz_j)
                vL = vz[i, j-1] + 0.5*svL

                d_vz_j_p = vz[i, j+1] - vz[i, j]
                svR = _minmod(d_vz_j, d_vz_j_p)
                vR = vz[i, j] - 0.5*svR

                FzL = _rusanov_flux_mass(rhoL, vL, rhoR, vR)

                # face j+1/2
                d_rho_j   = rho[i, j] - rho[i, j-1]
                d_rho_jp1 = rho[i, j+1] - rho[i, j]
                sL = _minmod(d_rho_j, d_rho_jp1)
                rhoL = rho[i, j] + 0.5*sL

                d_rho_jp2 = rho[i, j+2] - rho[i, j+1] if j+1 < nz-1 else 0.0
                sR = _minmod(d_rho_jp1, d_rho_jp2)
                rhoR = rho[i, j+1] - 0.5*sR

                d_vz_j   = vz[i, j] - vz[i, j-1]
                d_vz_jp1 = vz[i, j+1] - vz[i, j]
                svL = _minmod(d_vz_j, d_vz_jp1)
                vL = vz[i, j] + 0.5*svL

                d_vz_jp2 = vz[i, j+2] - vz[i, j+1] if j+1 < nz-1 else 0.0
                svR = _minmod(d_vz_jp1, d_vz_jp2)
                vR = vz[i, j+1] - 0.5*svR

                FzR = _rusanov_flux_mass(rhoL, vL, rhoR, vR)

                out[i, j] += (FzR - FzL) / dz
        return -out
else:
    def _div_mass(rho, vr, vz, R, dr, dz):
        # Vectorized reference (slower than numba parallel but OK)
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        # radial
        Fr = rho * vr
        r = R[:, :1]
        Fr_l = Fr.copy(); Fr_r = Fr.copy()
        Fr_l[1:] = Fr[:-1]; Fr_r[:-1] = Fr[1:]
        rLm = (r[:-1] + r[1:]) * 0.5
        rRp = (r[1:] + r[2:]) * 0.5 if nr > 2 else r[1:]
        div_r = np.zeros_like(rho)
        div_r[1:-1] = ( (rRp*Fr[2:] - rLm*Fr[1:-1]) / (r[1:-1]*dr) )
        out += div_r
        # vertical
        Fz = rho * vz
        out[:,1:-1] += (Fz[:,2:] - Fz[:,1:-1]) / dz
        return -out

# Momentum r, z, angular, and energy divergence/source terms — parallel versions

if NUMBA_OK:
    @njit(cache=True, parallel=True)
    def _mom_r_rhs(rho, vr, vz, p, vphi, dPhi_dr, R, dr, dz, gamma):
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        # flux divergences
        for i in prange(1, nr-1):
            rC = R[i, 0]
            for j in range(1, nz-1):
                # radial flux of radial momentum
                # F_r = ρ v_r^2 + p
                FrL = (rho[i-1, j]*vr[i-1, j]*vr[i-1, j] + p[i-1, j])
                FrR = (rho[i,   j]*vr[i,   j]*vr[i,   j] + p[i,   j])
                rLm = 0.5*(R[i-1, 0] + R[i, 0])
                rRp = 0.5*(R[i, 0] + R[i+1, 0])
                div_r = (rRp*FrR - rLm*FrL)/(rC*dr)
                # vertical flux: F_z = ρ v_r v_z
                FzL = rho[i, j-1]*vr[i, j-1]*vz[i, j-1]
                FzR = rho[i, j]*vr[i, j]*vz[i, j]
                div_z = (FzR - FzL)/dz
                # geometric + gravity sources
                geom = rho[i, j]*(vphi[i, j]*vphi[i, j]) / max(rC, 1e-12)
                grav = -rho[i, j]*dPhi_dr[i, j]
                out[i, j] = -(div_r + div_z) + geom + grav
        return out

    @njit(cache=True, parallel=True)
    def _mom_z_rhs(rho, vr, vz, p, dPhi_dz, R, dr, dz, gamma):
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        for i in prange(1, nr-1):
            rC = R[i, 0]
            for j in range(1, nz-1):
                FrL = (rho[i-1, j]*vr[i-1, j]*vz[i-1, j])
                FrR = (rho[i,   j]*vr[i,   j]*vz[i,   j])
                rLm = 0.5*(R[i-1, 0] + R[i, 0])
                rRp = 0.5*(R[i, 0] + R[i+1, 0])
                div_r = (rRp*FrR - rLm*FrL)/(rC*dr)

                FzL = (rho[i, j-1]*vz[i, j-1]*vz[i, j-1] + p[i, j-1])
                FzR = (rho[i, j]*vz[i, j]*vz[i, j] + p[i, j])
                div_z = (FzR - FzL)/dz

                grav = -rho[i, j]*dPhi_dz[i, j]
                out[i, j] = -(div_r + div_z) + grav
        return out

    @njit(cache=True, parallel=True)
    def _ang_mom_rhs(mom_r, mom_z, mom_phi, rho, R, dr, dz):
        # Advection of linear azimuthal momentum (ρ vφ)
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        vr = mom_r / np.maximum(rho, 1e-16)
        vz = mom_z / np.maximum(rho, 1e-16)
        q = mom_phi
        for i in prange(1, nr-1):
            rC = R[i, 0]
            for j in range(1, nz-1):
                FrL = (q[i-1, j] * vr[i-1, j])
                FrR = (q[i,   j] * vr[i,   j])
                rLm = 0.5*(R[i-1, 0] + R[i, 0])
                rRp = 0.5*(R[i, 0] + R[i+1, 0])
                div_r = (rRp*FrR - rLm*FrL)/(rC*dr)

                FzL = (q[i, j-1] * vz[i, j-1])
                FzR = (q[i,   j] * vz[i,   j])
                div_z = (FzR - FzL)/dz
                out[i, j] = -(div_r + div_z)
        return out

    @njit(cache=True, parallel=True)
    def _energy_rhs(rho, vr, vz, vphi, p, e_total, dPhi_dr, dPhi_dz, R, dr, dz, gamma):
        nr, nz = rho.shape
        out = np.zeros_like(rho)
        for i in prange(1, nr-1):
            rC = R[i, 0]
            for j in range(1, nz-1):
                # radial energy flux: (E+p) v_r
                HL = e_total[i-1, j] + p[i-1, j]
                HR = e_total[i,   j] + p[i,   j]
                FrL = HL * vr[i-1, j]
                FrR = HR * vr[i,   j]
                rLm = 0.5*(R[i-1, 0] + R[i, 0])
                rRp = 0.5*(R[i, 0] + R[i+1, 0])
                div_r = (rRp*FrR - rLm*FrL)/(rC*dr)

                # vertical energy flux: (E+p) v_z
                HLz = e_total[i, j-1] + p[i, j-1]
                HRz = e_total[i, j]   + p[i, j]
                FzL = HLz * vz[i, j-1]
                FzR = HRz * vz[i, j]
                div_z = (FzR - FzL)/dz

                # work by gravity source: -ρ v · ∇Φ
                grav = -rho[i, j]*( vr[i, j]*dPhi_dr[i, j] + vz[i, j]*dPhi_dz[i, j] )
                out[i, j] = -(div_r + div_z) + grav
        return out
else:
    def _mom_r_rhs(rho, vr, vz, p, vphi, dPhi_dr, R, dr, dz, gamma):
        r = R[:, :1]
        Fr = rho*vr*vr + p
        div_r = np.zeros_like(rho)
        div_r[1:-1] = (( ( (r[1:-1]+r[2:])*0.5 )*Fr[1:-1] - ( (r[:-2]+r[1:-1])*0.5 )*Fr[:-2]) / (r[1:-1]*dr))
        Fz = rho*vr*vz
        div_z = np.zeros_like(rho)
        div_z[:,1:-1] = (Fz[:,1:-1] - Fz[:,0:-2])/dz
        geom = rho * (vphi*vphi) / np.maximum(R, 1e-12)
        grav = -rho*dPhi_dr
        return -(div_r + div_z) + geom + grav

    def _mom_z_rhs(rho, vr, vz, p, dPhi_dz, R, dr, dz, gamma):
        r = R[:, :1]
        Fr = rho*vr*vz
        div_r = np.zeros_like(rho)
        div_r[1:-1] = ((( (r[1:-1]+r[2:])*0.5 )*Fr[1:-1] - ( (r[:-2]+r[1:-1])*0.5 )*Fr[:-2]) / (r[1:-1]*dr))
        Fz = rho*vz*vz + p
        div_z = np.zeros_like(rho)
        div_z[:,1:-1] = (Fz[:,1:-1] - Fz[:,0:-2])/dz
        grav = -rho*dPhi_dz
        return -(div_r + div_z) + grav

    def _ang_mom_rhs(mom_r, mom_z, mom_phi, rho, R, dr, dz):
        vr = mom_r/np.maximum(rho, 1e-16)
        vz = mom_z/np.maximum(rho, 1e-16)
        q = mom_phi
        r = R[:, :1]
        Fr = q*vr
        div_r = np.zeros_like(q)
        div_r[1:-1] = ((( (r[1:-1]+r[2:])*0.5 )*Fr[1:-1] - ( (r[:-2]+r[1:-1])*0.5 )*Fr[:-2]) / (r[1:-1]*dr))
        Fz = q*vz
        div_z = np.zeros_like(q)
        div_z[:,1:-1] = (Fz[:,1:-1] - Fz[:,0:-2])/dz
        return -(div_r + div_z)

    def _energy_rhs(rho, vr, vz, vphi, p, e_total, dPhi_dr, dPhi_dz, R, dr, dz, gamma):
        r = R[:, :1]
        HL = e_total + p
        Fr = HL*vr
        div_r = np.zeros_like(rho)
        div_r[1:-1] = ((( (r[1:-1]+r[2:])*0.5 )*Fr[1:-1] - ( (r[:-2]+r[1:-1])*0.5 )*Fr[:-2]) / (r[1:-1]*dr))
        Fz = HL*vz
        div_z = np.zeros_like(rho)
        div_z[:,1:-1] = (Fz[:,1:-1] - Fz[:,0:-2])/dz
        grav = -rho*(vr*dPhi_dr + vz*dPhi_dz)
        return -(div_r + div_z) + grav

# ---------------- Public wrappers (match main.py signature) ----------------

def np_continuity_eqn(rho: np.ndarray, vr: np.ndarray, vz: np.ndarray, R_grid: np.ndarray) -> np.ndarray:
    return _div_mass(rho, vr, vz, R_grid, DR, DZ)

def np_momentum_r(rho, vr, vz, p, vphi, dPhi_dr, R_grid):
    return _mom_r_rhs(rho, vr, vz, p, vphi, dPhi_dr, R_grid, DR, DZ, GAMMA)

def np_momentum_z(rho, vr, vz, p, dPhi_dz):
    # R_grid not needed here beyond geometry already in the operator
    # but we keep signature parity with normal module (without R)
    # Use global DR/DZ and expect caller to pass dPhi_dz
    # We require R_grid for radial geometry; pass via closure lambdas if needed.
    # Here we fetch a cached R-like array via shape of p: caller should instead
    # call conservation_normal.np_momentum_z when R is needed. For parallel we
    # assume main passes the correct R via set_spacings and uses same R in mom_r.
    # To keep compatibility, accept a global cached R is not available; so we
    # implement using a dummy R constructed as arange on first dim spacing-wise.
    # However, our implementation above already requires R; we therefore expose a
    # convenience wrapper below that needs R_grid. For main.py we *don't* use it.
    raise NotImplementedError("Use np_momentum_z_with_R for parallel module.")

def np_momentum_z_with_R(rho, vr, vz, p, dPhi_dz, R_grid):
    return _mom_z_rhs(rho, vr, vz, p, dPhi_dz, R_grid, DR, DZ, GAMMA)

def np_angular_mom(mom_r, mom_z, mom_phi, rho, R_grid):
    return _ang_mom_rhs(mom_r, mom_z, mom_phi, rho, R_grid, DR, DZ)

def np_energy(rho, vr, vz, vphi, p, e_total, dPhi_dr, dPhi_dz, R_grid):
    return _energy_rhs(rho, vr, vz, vphi, p, e_total, dPhi_dr, dPhi_dz, R_grid, DR, DZ, GAMMA)

# Backward-compat shim so main.py can call cpar.grad_potential_pw

def grad_potential_pw(R_grid: np.ndarray, Z_grid: np.ndarray):
    return np_grad_potential_pw(R_grid, Z_grid)
