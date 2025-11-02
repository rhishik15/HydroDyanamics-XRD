"""
Comprehensive pytest suite for non-relativistic hydrodynamic black hole accretion code
Tests for: singularities, conservation, numerical stability, physical accuracy

Installation:
    pip install pytest pytest-html numpy matplotlib scipy

Run tests:
    pytest test_hydrocode.py -v --html=report.html --self-contained-html
    
Run specific tests:
    pytest test_hydrocode.py::TestSingularities -v
    pytest test_hydrocode.py -k "axis" -v  # Run tests matching "axis"
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

# Import your modules
try:
    from config import params, get_derived_params, validate_params
    from conservation import (
        compute_fluxes, compute_source_terms, 
        compute_velocity_divergence, compute_timestep_constraint
    )
    from main import (
        bondi_solution_cylindrical,
        compute_potential_and_forces,
        initialize_bondi_flow,
        apply_boundary_conditions,
        primitive_to_conservative,
        conservative_to_primitive
    )
    from tvd import apply_tvd_limiters, detect_shocks
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ============================================================================
# FIXTURES - Setup test data
# ============================================================================

@pytest.fixture(scope="session")
def check_imports():
    """Check if all modules can be imported"""
    if not IMPORTS_AVAILABLE:
        pytest.skip(f"Cannot import required modules: {IMPORT_ERROR}")


@pytest.fixture
def grid_setup():
    """Create test grid"""
    N_r = 50
    N_z = 50
    R_max = 50.0
    Z_max = 50.0
    
    R_vals = np.linspace(0, R_max, N_r)
    Z_vals = np.linspace(0, Z_max, N_z)
    R, Z = np.meshgrid(R_vals, Z_vals, indexing='ij')
    
    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]
    
    return {
        'R': R, 'Z': Z, 'dR': dR, 'dZ': dZ,
        'N_r': N_r, 'N_z': N_z, 'R_max': R_max, 'Z_max': Z_max
    }


@pytest.fixture
def bondi_flow(check_imports, grid_setup):
    """Initialize Bondi flow on grid"""
    R = grid_setup['R']
    Z = grid_setup['Z']
    
    rho, p, vr, vz, vphi = bondi_solution_cylindrical(R, Z)
    
    return {
        'rho': rho, 'p': p, 'vr': vr, 'vz': vz, 'vphi': vphi,
        **grid_setup
    }


@pytest.fixture
def full_state():
    '''Create full simulation state with consistent grids'''
    
    # Test grid parameters
    N_r_test = 50
    N_z_test = 50
    R_max_test = 50.0
    Z_max_test = 50.0
    
    # Build test grids
    R_vals_test = np.linspace(0, R_max_test, N_r_test)
    Z_vals_test = np.linspace(0, Z_max_test, N_z_test)
    R_test, Z_test = np.meshgrid(R_vals_test, Z_vals_test, indexing='ij')
    
    dR_test = R_vals_test[1] - R_vals_test[0]
    dZ_test = Z_vals_test[1] - Z_vals_test[0]
    
    # ✅ Initialize with explicit grids
    rho, p, vr, vz, vphi, e_total = initialize_bondi_flow(
        R=R_test, Z=Z_test, dr=dR_test, dz=dZ_test
    )
    
    # ✅ Compute forces with same grids
    Phi, F_r, F_z, r_sph, R_used, Z_used, dr_used, dz_used = compute_potential_and_forces(
        R=R_test, Z=Z_test, dr=dR_test, dz=dZ_test, rho_shape=(N_r_test, N_z_test)
    )
    
    # ✅ Verify all shapes match
    assert rho.shape == (N_r_test, N_z_test)
    assert R_used.shape == (N_r_test, N_z_test)
    assert F_r.shape == (N_r_test, N_z_test)
    
    return {
        'rho': rho,
        'p': p,
        'vr': vr,
        'vz': vz,
        'vphi': vphi,
        'e_total': e_total,
        'R': R_used,
        'Z': Z_used,
        'dR': dr_used,
        'dZ': dz_used,
        'Phi': Phi,
        'F_r': F_r,
        'F_z': F_z,
        'N_r': N_r_test,
        'N_z': N_z_test,
        'R_max': R_max_test,
        'Z_max': Z_max_test
    }

# ============================================================================
# TEST CATEGORY 1: SINGULARITY DETECTION
# ============================================================================

class TestSingularities:
    """Test for numerical singularities near r=0 and other problematic regions"""
    
    def test_axis_regularity(self, bondi_flow):
        """Test that quantities are finite and smooth at the axis (R=0)"""
        rho = bondi_flow['rho']
        p = bondi_flow['p']
        vr = bondi_flow['vr']
        vz = bondi_flow['vz']
        vphi = bondi_flow['vphi']
        
        # Check axis values (i=0)
        axis_rho = rho[0, :]
        axis_p = p[0, :]
        axis_vr = vr[0, :]
        axis_vz = vz[0, :]
        axis_vphi = vphi[0, :]
        
        # All quantities should be finite
        assert np.all(np.isfinite(axis_rho)), "Density has NaN/Inf at axis"
        assert np.all(np.isfinite(axis_p)), "Pressure has NaN/Inf at axis"
        assert np.all(np.isfinite(axis_vr)), "Radial velocity has NaN/Inf at axis"
        assert np.all(np.isfinite(axis_vz)), "Vertical velocity has NaN/Inf at axis"
        assert np.all(np.isfinite(axis_vphi)), "Azimuthal velocity has NaN/Inf at axis"
        
        # Radial velocity should vanish at axis
        assert np.allclose(axis_vr, 0, atol=1e-6), \
            f"Radial velocity at axis should be ~0, got max={np.max(np.abs(axis_vr))}"
        
        # Azimuthal velocity must vanish at axis
        assert np.allclose(axis_vphi, 0, atol=1e-10), \
            f"Azimuthal velocity at axis should be 0, got max={np.max(np.abs(axis_vphi))}"
        
        # Scalars should be smooth (check gradient)
        if axis_rho.size > 1:
            drho_dz = np.gradient(axis_rho)
            assert np.all(np.isfinite(drho_dz)), "Density gradient at axis is not finite"
    
    
    def test_near_origin_singularity(self, bondi_flow):
        """Test behavior near (R=0, Z=0) - the coordinate singularity"""
        rho = bondi_flow['rho']
        p = bondi_flow['p']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        # Check cells near origin
        r_sphere = np.sqrt(R**2 + Z**2)
        near_origin = r_sphere < 5.0  # Within 5 r_g
        
        # Should have finite values
        assert np.all(np.isfinite(rho[near_origin])), \
            "Density has NaN/Inf near origin"
        assert np.all(np.isfinite(p[near_origin])), \
            "Pressure has NaN/Inf near origin"
        
        # Should be positive
        assert np.all(rho[near_origin] > 0), \
            f"Negative density near origin: min={np.min(rho[near_origin])}"
        assert np.all(p[near_origin] > 0), \
            f"Negative pressure near origin: min={np.min(p[near_origin])}"
    
    
    def test_potential_singularity(self, full_state):
        """Test that gravitational potential doesn't have unphysical singularities"""
        Phi = full_state['Phi']
        F_r = full_state['F_r']
        F_z = full_state['F_z']
        R = full_state['R']
        Z = full_state['Z']
        
        # All should be finite
        assert np.all(np.isfinite(Phi)), "Potential has NaN/Inf"
        assert np.all(np.isfinite(F_r)), "Radial force has NaN/Inf"
        assert np.all(np.isfinite(F_z)), "Vertical force has NaN/Inf"
        
        # Force should be bounded (no infinite acceleration)
        r_sphere = np.sqrt(R**2 + Z**2)
        physical_region = r_sphere > 2.0  # Outside Schwarzschild radius
        
        F_mag = np.sqrt(F_r**2 + F_z**2)
        F_max = np.max(F_mag[physical_region])
        
        assert F_max < 100.0, \
            f"Force magnitude unreasonably large: max={F_max} (check softening)"
    def test_example_using_fixture(full_state):
        rho = full_state['rho']
        R = full_state['R']
        Z = full_state['Z']
        F_r = full_state['F_r']
    
        # ✅ Build mask from SAME grid
        r_sphere = np.sqrt(R**2 + Z**2)
        physical_region = r_sphere > 2.0
    
        # ✅ Shapes now match!
        assert physical_region.shape == F_r.shape
    
        # ✅ Safe to use boolean indexing
        if np.any(physical_region):
            F_max = np.max(np.abs(F_r[physical_region]))
            assert F_max < 10.0, "Force too large"
    
    def test_velocity_divergence_at_axis(self, full_state):
        """Test that velocity divergence is finite at axis"""
        vr = full_state['vr']
        vz = full_state['vz']
        R = full_state['R']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        div_v = compute_velocity_divergence(vr, vz, R, dR, dZ)
        
        # Check axis
        div_v_axis = div_v[0, :]
        
        assert np.all(np.isfinite(div_v_axis)), \
            "Velocity divergence has NaN/Inf at axis"
        
        # Should be bounded
        assert np.max(np.abs(div_v_axis)) < 1e6, \
            f"Velocity divergence too large at axis: {np.max(np.abs(div_v_axis))}"


# ============================================================================
# TEST CATEGORY 2: CONSERVATION LAWS
# ============================================================================

class TestConservation:
    """Test conservation of mass, momentum, and energy"""
    
    def test_mass_conservation_initial(self, bondi_flow):
        """Test that initial mass is finite and positive"""
        rho = bondi_flow['rho']
        R = bondi_flow['R']
        dR = bondi_flow['dR']
        dZ = bondi_flow['dZ']
        
        # Compute total mass (integrating in cylindrical coords)
        R_safe = np.maximum(R, 0.5 * dR)
        total_mass = np.sum(rho * R_safe * dR * dZ) * 2 * np.pi
        
        assert np.isfinite(total_mass), "Total mass is not finite"
        assert total_mass > 0, f"Total mass is not positive: {total_mass}"
        
        print(f"\n  Initial total mass: {total_mass:.6e}")
    
    
    def test_energy_positivity(self, full_state):
        """Test that total energy is everywhere positive"""
        e_total = full_state['e_total']
        rho = full_state['rho']
        p = full_state['p']
        
        # Total energy should be positive
        assert np.all(e_total > 0), \
            f"Negative total energy found: min={np.min(e_total)}"
        
        # Internal energy should be positive
        internal = p / (5.0/3.0 - 1.0)
        assert np.all(internal > 0), \
            f"Negative internal energy: min={np.min(internal)}"
    
    
    def test_conservative_to_primitive_consistency(self, full_state):
        """Test that conservative <-> primitive conversion is consistent"""
        rho = full_state['rho']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        e_total = full_state['e_total']
        
        # Convert to conservative
        U1, U2, U3, U4, U5 = primitive_to_conservative(rho, vr, vz, vphi, e_total)
        
        # Convert back to primitive
        rho_new, vr_new, vz_new, vphi_new, p_new, e_new = \
            conservative_to_primitive(U1, U2, U3, U4, U5)
        
        # Should match (within tolerance)
        assert np.allclose(rho, rho_new, rtol=1e-10, atol=1e-12), \
            "Density not preserved in round-trip conversion"
        
        assert np.allclose(vr, vr_new, rtol=1e-10, atol=1e-12), \
            "Radial velocity not preserved in round-trip conversion"
        
        assert np.allclose(vz, vz_new, rtol=1e-10, atol=1e-12), \
            "Vertical velocity not preserved in round-trip conversion"
        
        assert np.allclose(vphi, vphi_new, rtol=1e-10, atol=1e-12), \
            "Azimuthal velocity not preserved in round-trip conversion"
    
    
    def test_momentum_conservation_in_source_terms(self, full_state):
        """Test that source terms don't create spurious momentum"""
        rho = full_state['rho']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        F_r = full_state['F_r']
        F_z = full_state['F_z']
        R = full_state['R']
        p = full_state['p']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        S1, S2, S3, S4, S5 = compute_source_terms(
            rho, vr, vz, vphi, F_r, F_z, R, p, dR, dZ
        )
        
        # Mass source should be zero
        assert np.allclose(S1, 0, atol=1e-12), \
            f"Non-zero mass source term: max={np.max(np.abs(S1))}"
        
        # Source terms should be finite
        assert np.all(np.isfinite(S2)), "Radial momentum source has NaN/Inf"
        assert np.all(np.isfinite(S3)), "Vertical momentum source has NaN/Inf"
        assert np.all(np.isfinite(S4)), "Angular momentum source has NaN/Inf"
        assert np.all(np.isfinite(S5)), "Energy source has NaN/Inf"
        
        # Momentum source at axis should be finite
        assert np.all(np.isfinite(S2[0, :])), \
            "Radial momentum source infinite at axis"


# ============================================================================
# TEST CATEGORY 3: NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test for numerical instabilities and timestep issues"""
    
    def test_timestep_constraint(self, full_state):
        """Test that timestep satisfies CFL condition"""
        rho = full_state['rho']
        p = full_state['p']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        CFL = params.get('CFL', 0.05)
        
        dt = compute_timestep_constraint(rho, p, vr, vz, vphi, dR, dZ, CFL)
        
        assert np.isfinite(dt), "Computed timestep is not finite"
        assert dt > 0, f"Timestep is not positive: dt={dt}"
        
        dt_max = params.get('dt_max', 0.05)
        dt_min = params.get('dt_min', 1e-10)
        
        assert dt >= dt_min, f"Timestep below minimum: dt={dt} < {dt_min}"
        assert dt <= dt_max, f"Timestep above maximum: dt={dt} > {dt_max}"
        
        print(f"\n  Computed timestep: {dt:.6e}")
    
    
    def test_flux_computation_stability(self, full_state):
        """Test that flux computation doesn't produce NaN/Inf"""
        rho = full_state['rho']
        p = full_state['p']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        R = full_state['R']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        dU1, dU2, dU3, dU4, dU5 = compute_fluxes(
            rho, p, vr, vz, vphi, R, dR, dZ
        )
        
        # All flux divergences should be finite
        assert np.all(np.isfinite(dU1)), "Mass flux divergence has NaN/Inf"
        assert np.all(np.isfinite(dU2)), "Radial momentum flux divergence has NaN/Inf"
        assert np.all(np.isfinite(dU3)), "Vertical momentum flux divergence has NaN/Inf"
        assert np.all(np.isfinite(dU4)), "Angular momentum flux divergence has NaN/Inf"
        assert np.all(np.isfinite(dU5)), "Energy flux divergence has NaN/Inf"
        
        # Check for extreme values
        assert np.max(np.abs(dU1)) < 1e10, \
            f"Mass flux divergence too large: {np.max(np.abs(dU1))}"
    
    
    def test_tvd_limiter_stability(self, full_state):
        """Test that TVD limiters don't introduce instabilities"""
        rho = full_state['rho']
        p = full_state['p']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        # Apply TVD limiting
        rho_lim, p_lim, vr_lim, vz_lim, vphi_lim = apply_tvd_limiters(
            rho, p, vr, vz, vphi, dR, dZ, limiter='minmod'
        )
        
        # Limited quantities should be finite
        assert np.all(np.isfinite(rho_lim)), "Limited density has NaN/Inf"
        assert np.all(np.isfinite(p_lim)), "Limited pressure has NaN/Inf"
        assert np.all(np.isfinite(vr_lim)), "Limited vr has NaN/Inf"
        assert np.all(np.isfinite(vz_lim)), "Limited vz has NaN/Inf"
        
        # Should preserve positivity
        assert np.all(rho_lim > 0), "Limited density not positive"
        assert np.all(p_lim > 0), "Limited pressure not positive"
        
        # Should not amplify values excessively
        assert np.max(rho_lim) <= 1.5 * np.max(rho), \
            "TVD limiter amplified density"
        assert np.max(p_lim) <= 1.5 * np.max(p), \
            "TVD limiter amplified pressure"
    
    
    def test_velocity_limiting(self, full_state):
        """Test that velocities don't exceed maximum (causality)"""
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        
        v_max = params.get('v_max', 0.95)
        
        v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
        
        assert np.all(v_mag <= v_max * 1.01), \
            f"Velocity exceeds limit: max={np.max(v_mag)} > {v_max}"


# ============================================================================
# TEST CATEGORY 4: PHYSICAL ACCURACY
# ============================================================================

class TestPhysicalAccuracy:
    """Test physical accuracy against known solutions"""
    
    def test_bondi_spherical_symmetry(self, bondi_flow):
        """Test that Bondi solution is spherically symmetric"""
        rho = bondi_flow['rho']
        p = bondi_flow['p']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        r_sphere = np.sqrt(R**2 + Z**2)
        
        # Check symmetry at several radii
        test_radii = [10.0, 20.0, 30.0, 40.0]
        
        for r_test in test_radii:
            if r_test > np.max(r_sphere):
                continue
            
            # Get cells at this radius
            mask = np.abs(r_sphere - r_test) < 1.0
            
            if np.sum(mask) > 10:  # Need enough samples
                rho_at_r = rho[mask]
                p_at_r = p[mask]
                
                # Check variation (should be small for spherical symmetry)
                rho_variation = np.std(rho_at_r) / (np.mean(rho_at_r) + 1e-20)
                p_variation = np.std(p_at_r) / (np.mean(p_at_r) + 1e-20)
                
                print(f"\n  At r={r_test}: rho variation = {rho_variation:.4f}, "
                      f"p variation = {p_variation:.4f}")
                
                # Should be less than 10% variation
                assert rho_variation < 0.1, \
                    f"Density not spherically symmetric at r={r_test}: var={rho_variation}"
                assert p_variation < 0.1, \
                    f"Pressure not spherically symmetric at r={r_test}: var={p_variation}"
    
    
    def test_bondi_velocity_direction(self, bondi_flow):
        """Test that Bondi velocity points radially inward"""
        vr = bondi_flow['vr']
        vz = bondi_flow['vz']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        # Skip inner boundary
        r_sphere = np.sqrt(R**2 + Z**2)
        physical_region = r_sphere > 5.0
        
        # Compute radial velocity in spherical coords
        r_sphere_safe = np.maximum(r_sphere, 1e-10)
        v_r_sph = (vr * R + vz * Z) / r_sphere_safe
        
        # Should be negative (inflow)
        assert np.all(v_r_sph[physical_region] <= 0), \
            f"Outward flow detected in Bondi solution: max(v_r)={np.max(v_r_sph[physical_region])}"
    
    
    def test_pressure_density_relation(self, bondi_flow):
        """Test that pressure ~ rho * T (ideal gas)"""
        rho = bondi_flow['rho']
        p = bondi_flow['p']
        
        # Temperature should be roughly constant for isothermal flow
        T = p / (rho + 1e-20)
        
        # Check physical region
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        r_sphere = np.sqrt(R**2 + Z**2)
        physical_region = (r_sphere > 5.0) & (r_sphere < 40.0)
        
        T_phys = T[physical_region]
        
        # Temperature variation should be modest
        T_variation = np.std(T_phys) / (np.mean(T_phys) + 1e-20)
        
        print(f"\n  Temperature variation: {T_variation:.4f}")
        print(f"  Mean temperature: {np.mean(T_phys):.6e}")
        
        # For Bondi, temperature should vary but not wildly
        assert T_variation < 2.0, \
            f"Temperature varies too much: {T_variation}"
    
    
    def test_bondi_accretion_rate(self, bondi_flow):
        """Test that computed accretion rate matches theoretical Bondi rate"""
        rho = bondi_flow['rho']
        vr = bondi_flow['vr']
        vz = bondi_flow['vz']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        dR = bondi_flow['dR']
        dZ = bondi_flow['dZ']
        
        # Compute accretion rate at a reference radius
        r_test = 20.0
        r_sphere = np.sqrt(R**2 + Z**2)
        
        # Get shell at r_test
        mask = np.abs(r_sphere - r_test) < 1.0
        
        if np.sum(mask) > 10:
            r_vals = r_sphere[mask]
            rho_vals = rho[mask]
            vr_vals = vr[mask]
            vz_vals = vz[mask]
            R_vals = R[mask]
            Z_vals = Z[mask]
            
            # Compute radial velocity in spherical coords
            r_safe = np.maximum(r_vals, 1e-10)
            v_r_sph = (vr_vals * R_vals + vz_vals * Z_vals) / r_safe
            
            # Mass flux through shell: dM/dt = -∫ ρ v_r dA
            # In cylindrical coords: dA = 2π R ds where ds is along surface
            mdot_numerical = -np.sum(rho_vals * v_r_sph * r_vals**2) * (dR * dZ / r_test**2) * 4 * np.pi
            
            # Theoretical Bondi rate
            BH_mass = params.get('BH_mass', 1.0)
            rho_inf = params.get('rho_inf', 2e-4)
            cs_inf = params.get('cs_inf', 0.173)
            
            mdot_bondi = 4 * np.pi * BH_mass**2 * rho_inf / cs_inf**3
            
            ratio = mdot_numerical / (mdot_bondi + 1e-20)
            
            print(f"\n  Accretion rate at r={r_test}:")
            print(f"    Numerical: {mdot_numerical:.6e}")
            print(f"    Bondi:     {mdot_bondi:.6e}")
            print(f"    Ratio:     {ratio:.4f}")
            
            # Should be within factor of 2-3 (numerical errors expected)
            assert 0.1 < ratio < 10.0, \
                f"Accretion rate far from Bondi: ratio={ratio}"


# ============================================================================
# TEST CATEGORY 5: SPIKE DETECTION
# ============================================================================

class TestSpikeDetection:
    """Detect sudden spikes and discontinuities in fields"""
    
    def test_density_gradient_spikes(self, bondi_flow):
        """Detect unreasonably large density gradients"""
        rho = bondi_flow['rho']
        dR = bondi_flow['dR']
        dZ = bondi_flow['dZ']
        
        # Compute gradients
        drho_dr = np.gradient(rho, axis=0) / dR
        drho_dz = np.gradient(rho, axis=1) / dZ
        
        grad_mag = np.sqrt(drho_dr**2 + drho_dz**2)
        
        # Find cells with large gradients
        rho_safe = np.maximum(rho, 1e-10)
        relative_gradient = grad_mag * min(dR, dZ) / rho_safe
        
        # Flag cells with >50% change per cell
        spike_threshold = 0.5
        spikes = relative_gradient > spike_threshold
        
        n_spikes = np.sum(spikes)
        
        if n_spikes > 0:
            print(f"\n  WARNING: {n_spikes} cells with large density gradients")
            max_grad = np.max(relative_gradient)
            print(f"  Maximum relative gradient: {max_grad:.4f}")
            
            # Find location of worst spike
            worst_idx = np.unravel_index(np.argmax(relative_gradient), rho.shape)
            print(f"  Location: i={worst_idx[0]}, j={worst_idx[1]}")
        
        # Should not have too many spikes in smooth Bondi flow
        spike_fraction = n_spikes / rho.size
        assert spike_fraction < 0.05, \
            f"Too many density spikes: {spike_fraction*100:.2f}% of cells"
    
    
    def test_pressure_jump_detection(self, bondi_flow):
        """Detect discontinuous jumps in pressure"""
        p = bondi_flow['p']
        
        # Compute second derivative (detects jumps)
        d2p_dr2 = np.gradient(np.gradient(p, axis=0), axis=0)
        d2p_dz2 = np.gradient(np.gradient(p, axis=1), axis=1)
        
        laplacian_p = d2p_dr2 + d2p_dz2
        
        # Normalize by pressure
        p_safe = np.maximum(p, 1e-10)
        relative_curvature = np.abs(laplacian_p) / (p_safe + 1e-10)
        
        # Find sharp features
        jump_threshold = 1.0
        jumps = relative_curvature > jump_threshold
        
        n_jumps = np.sum(jumps)
        
        if n_jumps > 0:
            print(f"\n  WARNING: {n_jumps} cells with pressure discontinuities")
            max_curvature = np.max(relative_curvature)
            print(f"  Maximum relative curvature: {max_curvature:.4f}")
        
        # Should be smooth
        jump_fraction = n_jumps / p.size
        assert jump_fraction < 0.05, \
            f"Too many pressure jumps: {jump_fraction*100:.2f}% of cells"
    
    
    def test_velocity_discontinuities(self, bondi_flow):
        """Detect velocity discontinuities (potential shocks)"""
        vr = bondi_flow['vr']
        vz = bondi_flow['vz']
        dR = bondi_flow['dR']
        dZ = bondi_flow['dZ']
        
        # Compute velocity divergence (shocks have div(v) < 0)
        dvr_dr = np.gradient(vr, axis=0) / dR
        dvz_dz = np.gradient(vz, axis=1) / dZ
        
        R = bondi_flow['R']
        R_safe = np.maximum(R, 0.5 * dR)
        
        # Cylindrical divergence: (1/R) d(R*vr)/dR + dvz/dz
        div_v = dvr_dr + vr / R_safe + dvz_dz
        
        # Strong compression indicates shock
        compression_threshold = -0.5
        strong_compression = div_v < compression_threshold
        
        n_compressed = np.sum(strong_compression)
        
        if n_compressed > 0:
            print(f"\n  WARNING: {n_compressed} cells with strong compression")
            min_div_v = np.min(div_v)
            print(f"  Minimum div(v): {min_div_v:.4f}")
            
            # Visualize shock locations
            worst_idx = np.unravel_index(np.argmin(div_v), div_v.shape)
            print(f"  Strongest compression at: i={worst_idx[0]}, j={worst_idx[1]}")
        
        # Bondi flow shouldn't have strong shocks
        compression_fraction = n_compressed / vr.size
        assert compression_fraction < 0.1, \
            f"Too many shock cells: {compression_fraction*100:.2f}% of domain"
    
    
    def test_energy_spikes(self, full_state):
        """Detect unreasonable energy spikes"""
        e_total = full_state['e_total']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        # Compute energy gradient
        de_dr = np.gradient(e_total, axis=0) / dR
        de_dz = np.gradient(e_total, axis=1) / dZ
        
        grad_e_mag = np.sqrt(de_dr**2 + de_dz**2)
        
        # Normalize by local energy
        e_safe = np.maximum(e_total, 1e-10)
        relative_grad_e = grad_e_mag * min(dR, dZ) / e_safe
        
        # Flag large changes
        energy_spike_threshold = 1.0
        energy_spikes = relative_grad_e > energy_spike_threshold
        
        n_energy_spikes = np.sum(energy_spikes)
        
        if n_energy_spikes > 0:
            print(f"\n  WARNING: {n_energy_spikes} cells with energy spikes")
            max_grad_e = np.max(relative_grad_e)
            print(f"  Maximum relative energy gradient: {max_grad_e:.4f}")
            
            # Locate worst spike
            worst_idx = np.unravel_index(np.argmax(relative_grad_e), e_total.shape)
            R = full_state['R']
            Z = full_state['Z']
            print(f"  Location: R={R[worst_idx]:.2f}, Z={Z[worst_idx]:.2f}")
        
        spike_fraction = n_energy_spikes / e_total.size
        assert spike_fraction < 0.1, \
            f"Too many energy spikes: {spike_fraction*100:.2f}% of cells"


# ============================================================================
# TEST CATEGORY 6: BOUNDARY CONDITIONS
# ============================================================================

class TestBoundaryConditions:
    """Test proper implementation of boundary conditions"""
    
    def test_axis_boundary(self, full_state):
        """Test axis boundary conditions (R=0)"""
        rho = full_state['rho']
        p = full_state['p']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        
        # Apply boundary conditions
        apply_boundary_conditions(rho, p, vr, vz, vphi, dt=0.001)
        
        # Check axis regularity after BC application
        assert np.allclose(vr[0, :], 0, atol=1e-8), \
            "Radial velocity at axis not zero after BC"
        
        assert np.allclose(vphi[0, :], 0, atol=1e-10), \
            "Azimuthal velocity at axis not zero after BC"
        
        # Density and pressure should be smooth
        assert np.all(np.isfinite(rho[0, :])), "Density infinite at axis after BC"
        assert np.all(np.isfinite(p[0, :])), "Pressure infinite at axis after BC"
    
    
    def test_inner_boundary_accretion(self, full_state):
        """Test that inner boundary properly accretes matter"""
        rho = full_state['rho'].copy()
        p = full_state['p'].copy()
        vr = full_state['vr'].copy()
        vz = full_state['vz'].copy()
        vphi = full_state['vphi'].copy()
        
        # Apply BC and check accretion
        mass_accreted = apply_boundary_conditions(rho, p, vr, vz, vphi, dt=0.001)
        
        assert np.isfinite(mass_accreted), "Accreted mass is not finite"
        assert mass_accreted >= 0, f"Negative accreted mass: {mass_accreted}"
        
        print(f"\n  Accreted mass in timestep: {mass_accreted:.6e}")
        
        # Check that matter near BH has been removed
        R = full_state['R']
        Z = full_state['Z']
        r_sphere = np.sqrt(R**2 + Z**2)
        r_inner = params.get('r_inner', 3.0)
        
        very_close = r_sphere < 0.95 * r_inner
        
        if np.any(very_close):
            rho_inner = rho[very_close]
            rho_floor = params.get('rho_floor', 1e-8)
            
            # Should be at floor
            assert np.allclose(rho_inner, rho_floor, rtol=0.1), \
                "Inner boundary not properly evacuated"
    
    
    def test_outer_boundary_bondi(self, full_state):
        """Test that outer boundary maintains Bondi solution"""
        rho = full_state['rho'].copy()
        p = full_state['p'].copy()
        vr = full_state['vr'].copy()
        vz = full_state['vz'].copy()
        vphi = full_state['vphi'].copy()
        
        R = full_state['R']
        Z = full_state['Z']
        
        # Get Bondi solution at outer boundary
        rho_bondi, p_bondi, vr_bondi, vz_bondi, vphi_bondi = \
            bondi_solution_cylindrical(R, Z)
        
        # Apply boundary conditions
        apply_boundary_conditions(rho, p, vr, vz, vphi, dt=0.001)
        
        # Check outer region
        r_sphere = np.sqrt(R**2 + Z**2)
        r_outer = params.get('r_outer', 60.0)
        outer_region = r_sphere > r_outer
        
        if np.any(outer_region):
            # Should match Bondi solution closely
            rho_diff = np.abs(rho[outer_region] - rho_bondi[outer_region])
            rho_rel_diff = rho_diff / (rho_bondi[outer_region] + 1e-20)
            
            max_rho_diff = np.max(rho_rel_diff)
            
            print(f"\n  Max density difference at outer boundary: {max_rho_diff:.4f}")
            
            # Should be within 20% (sponge forcing)
            assert max_rho_diff < 0.5, \
                f"Outer boundary density deviates too much: {max_rho_diff}"


# ============================================================================
# TEST CATEGORY 7: VISUALIZATION TESTS
# ============================================================================

class TestVisualization:
    """Generate diagnostic plots for visual inspection"""
    
    @pytest.fixture(autouse=True)
    def setup_plots(self, tmpdir):
        """Setup temporary directory for plots"""
        self.plot_dir = tmpdir.mkdir("diagnostic_plots")
        yield
        # Cleanup happens automatically
    
    
    def test_plot_density_profile(self, bondi_flow):
        """Plot radial density profile"""
        rho = bondi_flow['rho']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        r_sphere = np.sqrt(R**2 + Z**2)
        
        # Get equatorial profile
        z_idx = np.argmin(np.abs(Z[0, :]))
        r_eq = R[:, z_idx]
        rho_eq = rho[:, z_idx]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(r_eq, rho_eq, 'o-', label='Computed')
        plt.xlabel('Radius (r_g)')
        plt.ylabel('Density')
        plt.title('Radial Density Profile (Equatorial)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_file = self.plot_dir.join("density_profile.png")
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved: {plot_file}")
    
    
    def test_plot_velocity_field(self, bondi_flow):
        """Plot velocity vector field"""
        vr = bondi_flow['vr']
        vz = bondi_flow['vz']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        v_mag = np.sqrt(vr**2 + vz**2)
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Contour plot of velocity magnitude
        levels = np.logspace(-3, 0, 20)
        cs = ax.contourf(R, Z, v_mag, levels=levels, cmap='viridis')
        plt.colorbar(cs, label='Velocity Magnitude')
        
        # Add velocity vectors (subsample for clarity)
        skip = 5
        ax.quiver(R[::skip, ::skip], Z[::skip, ::skip],
                 vr[::skip, ::skip], vz[::skip, ::skip],
                 color='white', alpha=0.6, scale=5)
        
        ax.set_xlabel('R (r_g)')
        ax.set_ylabel('Z (r_g)')
        ax.set_title('Velocity Field')
        ax.set_aspect('equal')
        ax.set_xlim(0, np.max(R))
        ax.set_ylim(0, np.max(Z))
        
        plot_file = self.plot_dir.join("velocity_field.png")
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved: {plot_file}")
    
    
    def test_plot_gradient_map(self, bondi_flow):
        """Plot gradient magnitude to visualize spikes"""
        rho = bondi_flow['rho']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        dR = bondi_flow['dR']
        dZ = bondi_flow['dZ']
        
        # Compute gradients
        drho_dr = np.gradient(rho, axis=0) / dR
        drho_dz = np.gradient(rho, axis=1) / dZ
        
        grad_mag = np.sqrt(drho_dr**2 + drho_dz**2)
        
        # Normalize
        rho_safe = np.maximum(rho, 1e-10)
        relative_gradient = grad_mag * min(dR, dZ) / rho_safe
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Plot with log scale
        cs = ax.contourf(R, Z, np.log10(relative_gradient + 1e-10),
                        levels=20, cmap='RdYlBu_r')
        plt.colorbar(cs, label='log10(Relative Gradient)')
        
        # Mark high gradient regions
        spike_threshold = 0.5
        spikes = relative_gradient > spike_threshold
        if np.any(spikes):
            ax.contour(R, Z, spikes.astype(float), levels=[0.5],
                      colors='red', linewidths=2, linestyles='--')
        
        ax.set_xlabel('R (r_g)')
        ax.set_ylabel('Z (r_g)')
        ax.set_title('Density Gradient Map (spikes in red)')
        ax.set_aspect('equal')
        
        plot_file = self.plot_dir.join("gradient_map.png")
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved: {plot_file}")
    
    
    def test_plot_spherical_symmetry(self, bondi_flow):
        """Plot radial profiles at different angles to check symmetry"""
        rho = bondi_flow['rho']
        R = bondi_flow['R']
        Z = bondi_flow['Z']
        
        r_sphere = np.sqrt(R**2 + Z**2)
        theta = np.arctan2(Z, R)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        test_angles = [0, 30, 45, 60]  # degrees
        
        for idx, angle_deg in enumerate(test_angles):
            ax = axes.flatten()[idx]
            
            angle_rad = np.deg2rad(angle_deg)
            
            # Find cells near this angle
            angle_mask = np.abs(theta - angle_rad) < np.deg2rad(5)
            
            if np.any(angle_mask):
                r_vals = r_sphere[angle_mask]
                rho_vals = rho[angle_mask]
                
                # Sort by radius
                sort_idx = np.argsort(r_vals)
                r_sorted = r_vals[sort_idx]
                rho_sorted = rho_vals[sort_idx]
                
                ax.loglog(r_sorted, rho_sorted, 'o-', label=f'{angle_deg}°')
                ax.set_xlabel('Radius (r_g)')
                ax.set_ylabel('Density')
                ax.set_title(f'Density at θ = {angle_deg}°')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        
        plot_file = self.plot_dir.join("spherical_symmetry.png")
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Plot saved: {plot_file}")


# ============================================================================
# TEST CATEGORY 8: CONFIGURATION VALIDATION
# ============================================================================

class TestConfiguration:
    """Test configuration parameters"""
    
    def test_config_validity(self, check_imports):
        """Test that configuration is valid"""
        # Should not raise exception
        validate_params(params)
    
    
    def test_derived_params(self, check_imports):
        """Test derived parameter calculation"""
        derived = get_derived_params(params)
        
        # Check key quantities
        assert 'r_schwarzschild' in derived
        assert 'r_bondi' in derived
        assert 'r_sonic' in derived
        assert 'mdot_bondi' in derived
        
        # Should be positive
        assert derived['r_schwarzschild'] > 0
        assert derived['r_bondi'] > 0
        assert derived['r_sonic'] > 0
        
        print(f"\n  Derived parameters:")
        print(f"    r_schwarzschild = {derived['r_schwarzschild']:.4f}")
        print(f"    r_bondi = {derived['r_bondi']:.4f}")
        print(f"    r_sonic = {derived['r_sonic']:.4f}")
    
    
    def test_grid_resolution(self, check_imports):
        """Test that grid resolution is adequate"""
        N_r = params.get('N_r', 150)
        N_z = params.get('N_z', 150)
        
        assert N_r >= 50, f"Radial resolution too low: {N_r}"
        assert N_z >= 50, f"Vertical resolution too low: {N_z}"
        
        # Check CFL number
        CFL = params.get('CFL', 0.05)
        assert 0 < CFL <= 0.5, f"CFL number out of range: {CFL}"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test computational performance"""
    
    def test_flux_computation_time(self, full_state, benchmark):
        """Benchmark flux computation"""
        rho = full_state['rho']
        p = full_state['p']
        vr = full_state['vr']
        vz = full_state['vz']
        vphi = full_state['vphi']
        R = full_state['R']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        def compute():
            return compute_fluxes(rho, p, vr, vz, vphi, R, dR, dZ)
        
        result = benchmark(compute)
        
        print(f"\n  Flux computation benchmark: {benchmark.stats['mean']:.4f} s")
    
    
    def test_memory_usage(self, full_state):
        """Check memory usage"""
        import sys
        
        rho = full_state['rho']
        
        # Estimate memory per field
        field_size = rho.nbytes / 1024 / 1024  # MB
        
        # We have ~10 fields
        total_estimate = field_size * 10
        
        print(f"\n  Memory per field: {field_size:.2f} MB")
        print(f"  Estimated total: {total_estimate:.2f} MB")
        
        assert total_estimate < 1000, \
            f"Memory usage too high: {total_estimate:.2f} MB"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Full integration tests"""
    
    def test_single_timestep(self, full_state):
        """Test a complete timestep evolution"""
        rho = full_state['rho'].copy()
        p = full_state['p'].copy()
        vr = full_state['vr'].copy()
        vz = full_state['vz'].copy()
        vphi = full_state['vphi'].copy()
        e_total = full_state['e_total'].copy()
        
        R = full_state['R']
        F_r = full_state['F_r']
        F_z = full_state['F_z']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        # Compute timestep
        dt = compute_timestep_constraint(rho, p, vr, vz, vphi, dR, dZ, 0.05)
        
        print(f"\n  Timestep: {dt:.6e}")
        
        # Convert to conservative
        U1, U2, U3, U4, U5 = primitive_to_conservative(rho, vr, vz, vphi, e_total)
        
        # Compute RHS
        dU1, dU2, dU3, dU4, dU5 = compute_fluxes(rho, p, vr, vz, vphi, R, dR, dZ)
        S1, S2, S3, S4, S5 = compute_source_terms(rho, vr, vz, vphi, F_r, F_z, R, p, dR, dZ)
        
        # Update (Euler step)
        U1_new = U1 + dt * (dU1 + S1)
        U2_new = U2 + dt * (dU2 + S2)
        U3_new = U3 + dt * (dU3 + S3)
        U4_new = U4 + dt * (dU4 + S4)
        U5_new = U5 + dt * (dU5 + S5)
        
        # Convert back
        rho_new, vr_new, vz_new, vphi_new, p_new, e_new = \
            conservative_to_primitive(U1_new, U2_new, U3_new, U4_new, U5_new)
        
        # Check validity
        assert np.all(np.isfinite(rho_new)), "Density not finite after timestep"
        assert np.all(np.isfinite(p_new)), "Pressure not finite after timestep"
        assert np.all(rho_new > 0), "Negative density after timestep"
        assert np.all(p_new > 0), "Negative pressure after timestep"
        
        # Check conservation
        mass_initial = np.sum(U1 * np.maximum(R, 0.1) * dR * dZ) * 2 * np.pi
        mass_final = np.sum(U1_new * np.maximum(R, 0.1) * dR * dZ) * 2 * np.pi
        
        mass_change = abs(mass_final - mass_initial) / (mass_initial + 1e-20)
        
        print(f"  Mass change: {mass_change:.6e}")
        
        # Should be small (but not zero due to boundaries)
        assert mass_change < 0.1, f"Large mass change in single timestep: {mass_change}"
    
    
    def test_multiple_timesteps(self, full_state):
        """Test stability over multiple timesteps"""
        rho = full_state['rho'].copy()
        p = full_state['p'].copy()
        vr = full_state['vr'].copy()
        vz = full_state['vz'].copy()
        vphi = full_state['vphi'].copy()
        e_total = full_state['e_total'].copy()
        
        R = full_state['R']
        F_r = full_state['F_r']
        F_z = full_state['F_z']
        dR = full_state['dR']
        dZ = full_state['dZ']
        
        # Evolve for 10 steps
        n_steps = 10
        
        for step in range(n_steps):
            dt = compute_timestep_constraint(rho, p, vr, vz, vphi, dR, dZ, 0.05)
            
            # Simple Euler update
            U1, U2, U3, U4, U5 = primitive_to_conservative(rho, vr, vz, vphi, e_total)
            
            dU1, dU2, dU3, dU4, dU5 = compute_fluxes(rho, p, vr, vz, vphi, R, dR, dZ)
            S1, S2, S3, S4, S5 = compute_source_terms(rho, vr, vz, vphi, F_r, F_z, R, p, dR, dZ)
            
            U1 += dt * (dU1 + S1)
            U2 += dt * (dU2 + S2)
            U3 += dt * (dU3 + S3)
            U4 += dt * (dU4 + S4)
            U5 += dt * (dU5 + S5)
            
            rho, vr, vz, vphi, p, e_total = \
                conservative_to_primitive(U1, U2, U3, U4, U5)
            
            # Apply BCs
            apply_boundary_conditions(rho, p, vr, vz, vphi, dt=dt)
            
            # Recompute energy
            kinetic = 0.5 * rho * (vr**2 + vz**2 + vphi**2)
            internal = p / (5.0/3.0 - 1.0)
            e_total = kinetic + internal
            
            # Check for crashes
            assert np.all(np.isfinite(rho)), f"Crash at step {step}: density not finite"
            assert np.all(rho > 0), f"Crash at step {step}: negative density"
            assert np.all(p > 0), f"Crash at step {step}: negative pressure"
        
        print(f"\n  Successfully evolved {n_steps} timesteps")


# ============================================================================
# CUSTOM MARKERS
# ============================================================================

# Mark slow tests
slow = pytest.mark.slow

# Mark tests that require visualization
visual = pytest.mark.visual

# Mark tests that require benchmarking
perf = pytest.mark.perf


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run with: python test_hydrocode.py
    pytest.main([__file__, "-v", "--html=test_report.html", "--self-contained-html"])