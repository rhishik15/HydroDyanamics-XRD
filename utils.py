# utils.py - Grid validation and shape consistency utilities
"""
Utility functions to ensure grid/array shape consistency across all modules.
This prevents IndexError crashes from shape mismatches between state arrays and coordinate grids.
"""

import numpy as np
from config import params


def validate_and_build_local_grids(rho_like, R=None, Z=None, dr=None, dz=None, 
                                   R_max=None, Z_max=None):
    """
    Ensure R and Z coordinate grids match rho_like.shape exactly.
    
    If R or Z is None or has mismatched shape, builds new local grids
    using provided spacing (dr, dz) or domain size (R_max, Z_max).
    
    Parameters:
    -----------
    rho_like : array-like
        Reference array whose shape (Nr, Nz) determines grid dimensions
    R, Z : array-like, optional
        Existing coordinate grids to validate
    dr, dz : float, optional
        Grid spacing (used if grids need rebuilding)
    R_max, Z_max : float, optional
        Domain extent (used if grids need rebuilding)
    
    Returns:
    --------
    R_local, Z_local : ndarray
        Validated coordinate grids with shape matching rho_like
    dr_local, dz_local : float
        Grid spacings
    
    Raises:
    -------
    ValueError : If grids cannot be built (missing required parameters)
    """
    Nr, Nz = rho_like.shape
    
    # Check if provided grids are valid
    grids_valid = (R is not None and Z is not None and 
                   R.shape == rho_like.shape and Z.shape == rho_like.shape)
    
    if not grids_valid:
        # Need to build new grids
        if dr is None or dz is None:
            # Try using domain extents
            if R_max is None:
                R_max = params.get("R_max", 50.0)
            if Z_max is None:
                Z_max = params.get("Z_max", 50.0)
            
            R_vals = np.linspace(0.0, R_max, Nr)
            Z_vals = np.linspace(0.0, Z_max, Nz)
            
            dr_local = R_vals[1] - R_vals[0] if Nr > 1 else R_max / Nr
            dz_local = Z_vals[1] - Z_vals[0] if Nz > 1 else Z_max / Nz
        else:
            # Use provided spacings
            R_vals = np.arange(Nr) * dr
            Z_vals = np.arange(Nz) * dz
            dr_local = dr
            dz_local = dz
        
        # Create meshgrid with 'ij' indexing (R varies along axis 0)
        R_local, Z_local = np.meshgrid(R_vals, Z_vals, indexing='ij')
        
    else:
        # Use provided grids, infer spacings if needed
        R_local = R
        Z_local = Z
        
        if dr is None:
            dr_local = (R[1, 0] - R[0, 0]) if Nr > 1 else 1.0
        else:
            dr_local = dr
        
        if dz is None:
            dz_local = (Z[0, 1] - Z[0, 0]) if Nz > 1 else 1.0
        else:
            dz_local = dz
    
    return R_local, Z_local, dr_local, dz_local


def assert_consistent_shapes(*arrays, context=""):
    """
    Assert all arrays have identical shapes.
    
    Parameters:
    -----------
    *arrays : array-like
        Arrays to check
    context : str
        Context string for error message
    
    Raises:
    -------
    ValueError : If shapes are inconsistent
    """
    if len(arrays) < 2:
        return
    
    shapes = [a.shape for a in arrays]
    
    if not all(s == shapes[0] for s in shapes):
        msg = f"Inconsistent array shapes"
        if context:
            msg += f" in {context}"
        msg += f": {shapes}"
        raise ValueError(msg)


def safe_boolean_mask(array, mask, default_value=None):
    """
    Safely apply boolean mask, checking shape consistency.
    
    Parameters:
    -----------
    array : ndarray
        Array to index
    mask : ndarray (bool)
        Boolean mask
    default_value : scalar, optional
        Value to return if mask is empty or incompatible
    
    Returns:
    --------
    ndarray or scalar
        Masked array, or default_value if mask invalid
    """
    if mask.shape != array.shape:
        if default_value is not None:
            return default_value
        raise ValueError(f"Mask shape {mask.shape} != array shape {array.shape}")
    
    if not np.any(mask):
        if default_value is not None:
            return default_value
        return np.array([])
    
    return array[mask]


def compute_spherical_radius(R, Z, regularization=None):
    """
    Compute spherical radius with optional regularization.
    
    Parameters:
    -----------
    R, Z : ndarray
        Cylindrical coordinates
    regularization : float, optional
        Regularization parameter (sqrt(R² + Z² + ε²))
    
    Returns:
    --------
    r_sph : ndarray
        Spherical radius
    """
    if regularization is not None and regularization > 0:
        return np.sqrt(R**2 + Z**2 + regularization**2)
    else:
        return np.sqrt(R**2 + Z**2)


def get_regularized_radius(R, Z, R_reg):
    """
    Get regularized cylindrical radius for axis singularity treatment.
    
    Parameters:
    -----------
    R, Z : ndarray
        Cylindrical coordinates
    R_reg : float
        Regularization scale
    
    Returns:
    --------
    r_safe : ndarray
        Regularized radius: sqrt(R² + R_reg²)
    """
    return np.sqrt(R**2 + R_reg**2)