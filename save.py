import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches
from matplotlib import ticker
from config import params

R_max, Z_max = params["R_max"], params["Z_max"]
# Save inside your project "figures" folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_astrophysical_contours(field, R, Z, title="Field", filename="field.png", 
                               log_scale=True, levels=20, show_black_hole=True, 
                               show_marginally_stable=True, cmap="plasma",
                               show_contour_labels=True, percentile_clip=True):
    """
    Create publication-quality contour plots with labeled contour lines
    
    Parameters:
    -----------
    field : array
        2D field to plot
    R, Z : array
        Coordinate grids
    title : str
        Plot title
    filename : str
        Output filename
    log_scale : bool
        Use logarithmic colorscale
    levels : int
        Number of contour levels
    show_black_hole : bool
        Show black hole circle
    show_marginally_stable : bool
        Show marginally stable orbit
    cmap : str
        Colormap name
    show_contour_labels : bool
        Show values on contour lines
    percentile_clip : bool
        Use percentile-based clipping for better dynamic range
    """
    # Clean the data
    field_clean = np.copy(field)
    
    # Handle infinities and NaNs
    mask_bad = ~np.isfinite(field_clean)
    if np.any(mask_bad):
        field_clean[mask_bad] = np.nanmin(field_clean[~mask_bad]) if np.any(~mask_bad) else 1e-20
        print(f"  Warning: {np.sum(mask_bad)} bad values in {title}")
    
    # ========== Improved colorbar range calculation ==========
    if log_scale:
        field_clean = np.maximum(field_clean, 1e-20)
        
        # Calculate physical range (excluding obvious floor values)
        field_physical = field_clean[field_clean > 1e-6]
        
        if len(field_physical) > 100:
            if percentile_clip:
                # Use percentiles to focus on actual variation
                vmin = np.percentile(field_physical, 2)   # Ignore bottom 2%
                vmax = np.percentile(field_physical, 98)  # Ignore top 2%
            else:
                vmin = np.percentile(field_physical, 1)
                vmax = np.percentile(field_physical, 99)
            
            # Ensure reasonable dynamic range
            if vmax / vmin > 1e8:
                vmin = vmax / 1e6
            
            # Safety check for uniform fields
            if vmax / vmin < 1.05:  # Less than 5% variation
                print(f"  WARNING: {title} is nearly uniform, forcing range")
                vmin = np.min(field_clean[field_clean > 1e-10])
                vmax = np.max(field_clean)
                if vmax / vmin < 1.05:
                    vmin = vmin * 0.8
                    vmax = vmax * 1.2
        else:
            vmin = np.min(field_clean)
            vmax = np.max(field_clean)
            
        print(f"  {title} - Range: [{vmin:.2e}, {vmax:.2e}] (raw: [{np.min(field_clean):.2e}, {np.max(field_clean):.2e}])")
    else:
        # Linear scale
        if percentile_clip:
            vmin = np.percentile(field_clean, 2)
            vmax = np.percentile(field_clean, 98)
        else:
            vmin = np.percentile(field_clean, 1)
            vmax = np.percentile(field_clean, 99)
        
        # Safety check for uniform fields
        if abs(vmax - vmin) < 1e-10:
            print(f"  WARNING: {title} is nearly uniform")
            vmin = np.min(field_clean)
            vmax = np.max(field_clean)
            if abs(vmax - vmin) < 1e-10:
                vmin = vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1
                vmax = vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1
    
    # Final safety: ensure vmin < vmax
    if vmin >= vmax:
        print(f"  ERROR: Invalid range for {title}, forcing separation")
        if vmin == 0:
            vmin = 0
            vmax = 1e-6
        else:
            vmax = vmin * 1.1
            vmin = vmin * 0.9
    
    # Clip the field to the chosen range
    field_plot = np.clip(field_clean, vmin, vmax)
    
    # --- Create the plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))
    
    # Create contour levels
    if log_scale and vmin > 0:
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)
        
        # Ensure log levels are valid
        if log_max - log_min < 1e-6:
            log_max = log_min + 0.1
        
        contour_levels = np.logspace(log_min, log_max, levels)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        # Ensure linear levels are valid
        if vmax - vmin < 1e-10:
            vmax = vmin + 1e-6
        
        contour_levels = np.linspace(vmin, vmax, levels)
        norm = None
    
    # Double-check contour levels are strictly increasing
    contour_levels = np.unique(contour_levels)
    if len(contour_levels) < 2:
        print(f"  ERROR: Cannot create valid contours for {title}, skipping")
        plt.close()
        return
    
    try:
        # Main contour plot (filled)
        cs_filled = ax.contourf(R, Z, field_plot, levels=contour_levels, 
                               cmap=cmap, norm=norm, extend='both')
        
        # ========== ENHANCED: Contour lines with labels ==========
        if len(contour_levels) >= 6:
            # Select subset of levels for labeled lines (every 3rd or 4th level)
            label_interval = max(3, len(contour_levels) // 8)  # Adaptive interval
            label_levels = contour_levels[::label_interval]
            
            # Draw contour lines
            cs_lines = ax.contour(R, Z, field_plot, levels=label_levels, 
                                 colors='white', alpha=0.6, linewidths=1.2,
                                 linestyles='solid')
            
            if show_contour_labels:
                # Add labels to contour lines with proper formatting
                if log_scale:
                    # For log scale, use lambda function for scientific notation
                    labels = ax.clabel(cs_lines, inline=True, inline_spacing=8, 
                                      fontsize=8, fmt=lambda x: f'{x:.2e}', 
                                      colors='black', use_clabeltext=True)
                else:
                    # For linear scale, show fixed decimals
                    labels = ax.clabel(cs_lines, inline=True, inline_spacing=8, 
                                      fontsize=8, fmt='%.2g', 
                                      colors='black', use_clabeltext=True)
                
                # Add white background boxes to labels for readability
                for txt in labels:
                    txt.set_bbox(dict(boxstyle='round,pad=0.4', 
                                     facecolor='white', alpha=0.85, 
                                     edgecolor='gray', linewidth=0.5))
                    txt.set_fontweight('bold')
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(cs_filled, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, rotation=270, labelpad=25, fontsize=13, fontweight='bold')
        
        # Improve colorbar tick labels
        if log_scale:
            cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        
        cbar.ax.tick_params(labelsize=10)
        
    except Exception as e:
        print(f"  ERROR creating contours for {title}: {e}")
        plt.close()
        return
    
    # Mark important astrophysical features
    if show_black_hole:
        bh_circle = patches.Circle((0, 0), 2.5, color='black', fill=True, 
                                  zorder=10, label='Black Hole ($r < 2.5 r_g$)')
        ax.add_patch(bh_circle)
    
    if show_marginally_stable:
        ms_circle = patches.Circle((0, 0), 6.0, color='cyan', fill=False, 
                                  linestyle='--', linewidth=2, alpha=0.8, 
                                  zorder=9, label='$r = 6 r_g$')
        ax.add_patch(ms_circle)
    
    # Add reference lines
    ax.axhline(y=0, color='white', linestyle=':', alpha=0.5, linewidth=1.5, 
              label='Equatorial Plane')
    ax.axvline(x=0, color='white', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('$R$ (Schwarzschild radii $r_g$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('$Z$ (Schwarzschild radii $r_g$)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, pad=20, fontweight='bold')
    ax.set_aspect('equal')
    
    # Set plot limits
    r_plot_max = R_max
    ax.set_xlim(0, r_plot_max)
    ax.set_ylim(0, r_plot_max)
   
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')
    ax.tick_params(labelsize=11)
    
    # Legend with better formatting
    if show_black_hole or show_marginally_stable:
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                          fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  → Saved: {filename}")


def plot_field_enhanced(field, R, Z, title="Field", cmap="inferno", filename="field.png", 
                       log_scale=False, vmin=None, vmax=None, show_contours=False):
    """
    Enhanced plotting with better handling of extreme values
    """
    plot_astrophysical_contours(field, R, Z, title=title, filename=filename, 
                               log_scale=log_scale, cmap=cmap)


def plot_comparative_contours(fields_dict, R, Z, filename="comparative_fields.png"):
    """
    Create multi-panel comparison plot with ALL fields and labeled contours
    """
    n_fields = len(fields_dict)
    if n_fields == 0:
        return
    
    print(f"\n  Creating comparison plot with {n_fields} fields...")
    
    # Determine optimal subplot layout
    if n_fields <= 2:
        rows, cols = 1, n_fields
        figsize = (12 * n_fields, 10)
    elif n_fields <= 4:
        rows, cols = 2, 2
        figsize = (20, 20)
    elif n_fields <= 6:
        rows, cols = 2, 3
        figsize = (30, 20)
    elif n_fields <= 9:
        rows, cols = 3, 3
        figsize = (30, 30)
    else:
        # For more than 9 fields, use 4 columns
        rows = int(np.ceil(n_fields / 4))
        cols = 4
        figsize = (40, 10 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_fields == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Color maps for different physical quantities
    cmaps = {
        'density': 'plasma',
        'pressure': 'viridis', 
        'temperature': 'hot',
        'energy': 'inferno',
        'velocity': 'RdBu_r',
        'mach': 'coolwarm',
        'sound': 'cool',
        'radial': 'seismic'
    }
    
    for idx, (field_name, field_data) in enumerate(fields_dict.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Clean data
        field_clean = np.copy(field_data)
        mask_bad = ~np.isfinite(field_clean)
        if np.any(mask_bad):
            field_clean[mask_bad] = np.nanmin(field_clean[~mask_bad]) if np.any(~mask_bad) else 1e-20
        
        # Choose appropriate colormap
        cmap = 'plasma'  # default
        for key, color in cmaps.items():
            if key.lower() in field_name.lower():
                cmap = color
                break
        
        # Determine if log scale is appropriate
        log_scale = (np.min(field_clean) > 0 and 
                    np.max(field_clean) / np.min(field_clean) > 100)
        
        # Calculate colorbar range with percentile clipping
        if log_scale:
            field_clean = np.maximum(field_clean, 1e-20)
            field_physical = field_clean[field_clean > 1e-6]
            
            if len(field_physical) > 100:
                vmin = np.percentile(field_physical, 2)
                vmax = np.percentile(field_physical, 98)
                if vmax / vmin > 1e8:
                    vmin = vmax / 1e6
            else:
                vmin = np.min(field_clean)
                vmax = np.max(field_clean)
            
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = np.percentile(field_clean, 2)
            vmax = np.percentile(field_clean, 98)
            if abs(vmax - vmin) < 1e-10:
                vmax = vmin + 1e-6
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Create contour levels
        levels = 15
        if log_scale:
            log_min, log_max = np.log10(vmin), np.log10(vmax)
            contour_levels = np.logspace(log_min, log_max, levels)
        else:
            contour_levels = np.linspace(vmin, vmax, levels)
        
        contour_levels = np.unique(contour_levels)
        
        # Filled contours
        cs = ax.contourf(R, Z, field_clean, levels=contour_levels, 
                        cmap=cmap, norm=norm, extend='both')
        
        # Line contours with labels
        label_levels = contour_levels[::max(2, len(contour_levels) // 6)]
        cs_lines = ax.contour(R, Z, field_clean, levels=label_levels, 
                             colors='white', alpha=0.6, linewidths=1.0)
        
        # Add contour labels with proper formatting
        if log_scale:
            # For log scale, use lambda for scientific notation
            labels = ax.clabel(cs_lines, inline=True, fontsize=7, 
                              fmt=lambda x: f'{x:.2e}', 
                              colors='black')
        else:
            # For linear scale, use simple format
            labels = ax.clabel(cs_lines, inline=True, fontsize=7, 
                              fmt='%.2g', 
                              colors='black')
        
        for txt in labels:
            txt.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.8, edgecolor='none'))
        
        # Add black hole and marginally stable orbit
        bh_circle = patches.Circle((0, 0), 2.5, color='black', fill=True, zorder=10)
        ms_circle = patches.Circle((0, 0), 6.0, color='cyan', fill=False, 
                                  linestyle='--', linewidth=1.5, alpha=0.7, zorder=9)
        ax.add_patch(bh_circle)
        ax.add_patch(ms_circle)
        
        # Add reference lines
        ax.axhline(y=0, color='white', linestyle=':', alpha=0.4, linewidth=1)
        ax.axvline(x=0, color='white', linestyle=':', alpha=0.4, linewidth=1)
        
        # Formatting
        ax.set_xlabel('$R$ ($r_g$)', fontsize=11, fontweight='bold')
        ax.set_ylabel('$Z$ ($r_g$)', fontsize=11, fontweight='bold')
        ax.set_title(field_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_aspect('equal')
        
        ax.set_xlim(np.min(R), np.max(R))
        ax.set_ylim(np.min(Z), np.max(Z))
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
        if log_scale:
            cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            cbar.set_label(f'log({field_name})', rotation=270, labelpad=15, fontsize=10)
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
            cbar.set_label(field_name, rotation=270, labelpad=15, fontsize=10)
        
        cbar.ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(len(fields_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  → Saved: {filename}")


def save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0):
    """
    Enhanced saving with publication-quality plots and labeled contours
    """
    print(f"\n{'='*70}")
    print(f"SAVING ASTROPHYSICAL DATA - STEP {step}")
    print(f"{'='*70}")
    
    # --- Basic thermodynamic quantities ---
    print("\n[1/12] Plotting Density...")
    plot_astrophysical_contours(rho, R, Z, title="Density ($\\rho$)", 
                               filename=f"density_{step:05d}.png", 
                               log_scale=True, cmap="plasma",
                               show_contour_labels=True)
    
    print("[2/12] Plotting Pressure...")
    plot_astrophysical_contours(p, R, Z, title="Pressure ($p$)", 
                               filename=f"pressure_{step:05d}.png", 
                               log_scale=True, cmap="viridis",
                               show_contour_labels=True)
    
    print("[3/12] Plotting Total Energy...")
    # Mask out the black hole interior
    e_plot = e_total.copy()
    
    r_inner = params["r_inner"]
    Rg = np.sqrt(R**2 + Z**2)
    e_plot[Rg < r_inner] = np.nan   # remove unphysical region inside inner radius

    plot_astrophysical_contours(e_plot, R, Z, title="Total Energy ($E$)", 
                               filename=f"energy_{step:05d}.png", 
                               log_scale=True, cmap="inferno",
                               show_contour_labels=True)
    
    # --- Temperature ---
    print("[4/12] Plotting Temperature...")
    T = p / (rho + 1e-20)
    plot_astrophysical_contours(T, R, Z, title="Temperature ($T = p/\\rho$)", 
                               filename=f"temperature_{step:05d}.png", 
                               log_scale=True, cmap="hot",
                               show_contour_labels=True)
    
    # --- Velocity components ---
    print("[5/12] Plotting Radial Velocity...")
    plot_astrophysical_contours(vr, R, Z, title="Radial Velocity ($v_R$)", 
                               filename=f"vr_{step:05d}.png", 
                               log_scale=False, cmap="RdBu_r",
                               show_contour_labels=True)
    
    print("[6/12] Plotting Vertical Velocity...")
    plot_astrophysical_contours(vz, R, Z, title="Vertical Velocity ($v_Z$)", 
                               filename=f"vz_{step:05d}.png", 
                               log_scale=False, cmap="RdBu_r",
                               show_contour_labels=True)
    
    print("[7/12] Plotting Velocity Magnitude...")
    v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
    plot_astrophysical_contours(v_mag, R, Z, title="Velocity Magnitude ($|\\mathbf{v}|$)", 
                               filename=f"vmag_{step:05d}.png", 
                               log_scale=False, cmap="magma",
                               show_contour_labels=True)
    
    # --- Azimuthal velocity (if present) ---
    if np.any(np.abs(vphi) > 1e-10):
        print("[8/12] Plotting Azimuthal Velocity...")
        plot_astrophysical_contours(vphi, R, Z, title="Azimuthal Velocity ($v_\\phi$)", 
                                   filename=f"vphi_{step:05d}.png", 
                                   log_scale=False, cmap="RdBu_r",
                                   show_contour_labels=True)
    else:
        print("[8/12] Skipping Azimuthal Velocity (negligible)...")
    
    # --- Derived quantities ---
    print("[9/12] Plotting Sound Speed...")
    cs = np.sqrt(np.maximum(5.0/3.0 * p / (rho + 1e-20), 1e-20))
    plot_astrophysical_contours(cs, R, Z, title="Sound Speed ($c_s$)", 
                               filename=f"sound_speed_{step:05d}.png", 
                               log_scale=True, cmap="cool",
                               show_contour_labels=True)
    
    print("[10/12] Plotting Radial Mach Number...")
    mach_r = np.abs(vr) / (cs + 1e-20)
    plot_astrophysical_contours(mach_r, R, Z, title="Radial Mach Number ($|v_R|/c_s$)", 
                               filename=f"mach_{step:05d}.png", 
                               log_scale=False, cmap="coolwarm",
                               show_contour_labels=True)
    
    # --- Specific angular momentum ---
    print("[11/12] Plotting Specific Angular Momentum...")
    l_spec = R * vphi
    if np.any(np.abs(l_spec) > 1e-10):
        plot_astrophysical_contours(l_spec, R, Z, 
                                   title="Specific Angular Momentum ($R v_\\phi$)", 
                                   filename=f"angular_momentum_{step:05d}.png", 
                                   log_scale=False, cmap="RdBu_r",
                                   show_contour_labels=True)
    else:
        print("  Skipping (negligible angular momentum)...")
    
    # --- ALWAYS create comparison plot ---
    print("[12/12] Creating Multi-Panel Comparison Plot...")
    comparison_fields = {
        'Density ($\\rho$)': rho,
        'Temperature ($T$)': T,
        'Pressure ($p$)': p,
        'Radial Velocity ($v_R$)': vr,
        'Velocity Magnitude ($|v|$)': v_mag,
        'Mach Number ($M$)': mach_r,
        'Sound Speed ($c_s$)': cs,
        'Energy ($E$)': e_total
    }
    
    plot_comparative_contours(comparison_fields, R, Z, 
                            filename=f"comparison_{step:05d}.png")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"DATA SUMMARY - STEP {step}")
    print(f"{'='*70}")
    print(f"  Density       : [{np.min(rho):.2e}, {np.max(rho):.2e}]")
    print(f"  Temperature   : [{np.min(T):.2e}, {np.max(T):.2e}]")
    print(f"  Pressure      : [{np.min(p):.2e}, {np.max(p):.2e}]")
    print(f"  Radial Vel    : [{np.min(vr):.3f}, {np.max(vr):.3f}]")
    print(f"  Velocity Mag  : [{np.min(v_mag):.3f}, {np.max(v_mag):.3f}]")
    print(f"  Mach Number   : [{np.min(mach_r):.3f}, {np.max(mach_r):.3f}]")
    print(f"  Sound Speed   : [{np.min(cs):.3f}, {np.max(cs):.3f}]")
    print(f"{'='*70}\n")


def save_all(rho, p, e_total, R, Z, step=0):
    """
    Backward compatibility function
    """
    vphi = np.zeros_like(rho)
    vr = np.zeros_like(rho)
    vz = np.zeros_like(rho)
    
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step)


def plot_radial_profiles(fields_dict, R, Z, radii=[5, 10, 20, 30], filename="radial_profiles.png"):
    """
    Plot radial profiles at different radii
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    Rg = np.sqrt(R**2 + Z**2)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    
    plot_idx = 0
    for field_name, field_data in fields_dict.items():
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        
        for i, radius in enumerate(radii):
            if radius > np.max(Rg):
                continue
                
            mask_radius = (np.abs(Rg - radius) < 1.0) & (Z >= 0)
            
            if np.any(mask_radius):
                theta_vals = np.arctan2(Z[mask_radius], R[mask_radius])
                field_vals = field_data[mask_radius]
                
                sort_idx = np.argsort(theta_vals)
                theta_sorted = theta_vals[sort_idx]
                field_sorted = field_vals[sort_idx]
                
                ax.semilogy(theta_sorted * 180/np.pi, np.abs(field_sorted), 
                           'o-', color=colors[i % len(colors)], 
                           label=f'$r = {radius} r_g$', markersize=4, linewidth=2)
        
        ax.set_xlabel("Polar Angle (degrees)", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"|{field_name}|", fontsize=12, fontweight='bold')
        ax.set_title(f"{field_name} vs Angle", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plot_idx += 1
    
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()


def plot_equatorial_profile(fields_dict, R, Z, filename="equatorial_profile.png"):
    """
    Plot profiles along equatorial plane
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    z_idx = np.argmin(np.abs(Z[0, :]))
    r_vals = R[:, z_idx]
    
    plot_idx = 0
    for field_name, field_data in fields_dict.items():
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        field_vals = field_data[:, z_idx]
        
        ax.loglog(r_vals, np.abs(field_vals), 'o-', linewidth=2.5, 
                 markersize=5, color='#377eb8')
        ax.set_xlabel("Radius ($r_g$)", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"|{field_name}|", fontsize=12, fontweight='bold')
        ax.set_title(f"{field_name} along Equatorial Plane", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        
        # Mark important radii
        ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='$r_{BH} = 2.5')
        ax.axvline(x=6, color='orange', linestyle=':', alpha=0.7, linewidth=2, 
                   label='$r = 6')
        ax.legend(fontsize=10, framealpha=0.9)
        
        plot_idx += 1
    
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()


def plot_velocity_field(vr, vz, R, Z, filename="velocity_field.png", skip=5):
    """
    Plot velocity vector field with streamlines
    
    Parameters:
    -----------
    vr, vz : array
        Velocity components
    R, Z : array
        Coordinate grids
    filename : str
        Output filename
    skip : int
        Plot every skip-th arrow for clarity
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))
    
    # Calculate velocity magnitude
    v_mag = np.sqrt(vr**2 + vz**2)
    
    # Plot velocity magnitude as background
    levels = 20
    vmin = np.percentile(v_mag, 2)
    vmax = np.percentile(v_mag, 98)
    
    contour_levels = np.linspace(vmin, vmax, levels)
    cs = ax.contourf(R, Z, v_mag, levels=contour_levels, cmap='viridis', alpha=0.7)
    
    # Add streamlines
    ax.streamplot(R, Z, vr, vz, color='white', linewidth=1.5, 
                  density=1.5, arrowsize=1.5, arrowstyle='->')
    
    # Add quiver plot (arrows) - subsample for clarity
    R_sub = R[::skip, ::skip]
    Z_sub = Z[::skip, ::skip]
    vr_sub = vr[::skip, ::skip]
    vz_sub = vz[::skip, ::skip]
    
    ax.quiver(R_sub, Z_sub, vr_sub, vz_sub, 
             color='cyan', alpha=0.6, scale=5, width=0.003)
    
    # Add black hole
    bh_circle = patches.Circle((0, 0), 2.5, color='black', fill=True, zorder=10)
    ax.add_patch(bh_circle)
    
    ms_circle = patches.Circle((0, 0), 6.0, color='red', fill=False, 
                              linestyle='--', linewidth=2, alpha=0.8, zorder=9)
    ax.add_patch(ms_circle)
    
    # Colorbar
    cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity Magnitude $|\\mathbf{v}|', rotation=270, 
                   labelpad=25, fontsize=13, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('$R$ ($r_g$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('$Z$ ($r_g$)', fontsize=13, fontweight='bold')
    ax.set_title('Velocity Field with Streamlines', fontsize=15, 
                 fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.set_xlim(0, np.max(R))
    ax.set_ylim(0, np.max(Z))
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()


def plot_thermodynamic_state(rho, p, T, filename="thermodynamic_state.png"):
    """
    Plot density-temperature phase diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    
    # Flatten arrays
    rho_flat = rho.flatten()
    T_flat = T.flatten()
    
    # Remove floor values for clearer visualization
    mask_physical = (rho_flat > 1e-7) & (T_flat > 1e-5)
    rho_phys = rho_flat[mask_physical]
    T_phys = T_flat[mask_physical]
    
    # 2D histogram
    hist, xedges, yedges = np.histogram2d(np.log10(rho_phys), np.log10(T_phys), 
                                          bins=50)
    
    # Plot
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(hist.T, origin='lower', extent=extent, aspect='auto', 
                   cmap='plasma', interpolation='bilinear')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Cells', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('$\\log_{10}(\\rho)', fontsize=13, fontweight='bold')
    ax.set_ylabel('$\\log_{10}(T)', fontsize=13, fontweight='bold')
    ax.set_title('Density-Temperature Phase Diagram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def create_animation(field_name="density", start_step=0, end_step=100, 
                    step_interval=10, fps=10):
    """
    Create animation from saved images
    
    Parameters:
    -----------
    field_name : str
        Name of field to animate
    start_step : int
        Starting step number
    end_step : int
        Ending step number
    step_interval : int
        Interval between frames
    fps : int
        Frames per second
    """
    try:
        import imageio
        import glob
        
        print(f"\nCreating animation for {field_name}...")
        
        # Find matching files
        pattern = os.path.join(OUTPUT_DIR, f"{field_name}_*.png")
        files = sorted(glob.glob(pattern))
        
        if len(files) < 2:
            print(f"  Not enough files found for {field_name} animation")
            return
        
        # Filter by step range
        filtered_files = []
        for f in files:
            try:
                step_num = int(f.split('_')[-1].split('.')[0])
                if start_step <= step_num <= end_step and step_num % step_interval == 0:
                    filtered_files.append(f)
            except:
                continue
        
        if len(filtered_files) < 2:
            print(f"  Not enough files in range for {field_name} animation")
            return
        
        print(f"  Found {len(filtered_files)} frames")
        
        # Create animation
        images = []
        for filename in filtered_files:
            images.append(imageio.imread(filename))
        
        output_file = os.path.join(OUTPUT_DIR, f"{field_name}_animation.gif")
        imageio.mimsave(output_file, images, duration=1.0/fps)
        print(f"  → Animation saved: {output_file}")
        
    except ImportError:
        print("  ERROR: imageio not available - install with: pip install imageio")
    except Exception as e:
        print(f"  ERROR creating animation: {e}")


def create_all_animations(start_step=0, end_step=100, step_interval=10):
    """
    Create animations for all standard fields
    """
    fields = ['density', 'temperature', 'pressure', 'energy', 
              'vr', 'vz', 'vmag', 'mach', 'sound_speed', 'comparison']
    
    print(f"\n{'='*70}")
    print("CREATING ANIMATIONS")
    print(f"{'='*70}")
    
    for field in fields:
        create_animation(field, start_step, end_step, step_interval)
    
    print(f"{'='*70}\n")


def cleanup_old_files(max_files=500):
    """
    Clean up old files to prevent disk space issues
    
    Parameters:
    -----------
    max_files : int
        Maximum number of files to keep
    """
    try:
        import glob
        
        print(f"\nCleaning up old files (keeping most recent {max_files})...")
        
        # Get all PNG files
        files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
        
        if len(files) > max_files:
            # Sort by modification time
            files.sort(key=os.path.getmtime)
            
            # Remove oldest files
            files_to_remove = files[:len(files) - max_files]
            
            for f in files_to_remove:
                os.remove(f)
            
            print(f"  Cleaned up {len(files_to_remove)} old files")
        else:
            print(f"  No cleanup needed ({len(files)} files < {max_files} limit)")
    
    except Exception as e:
        print(f"  ERROR during cleanup: {e}")


def save_raw_data(rho, p, e_total, vr, vz, vphi, R, Z, step=0):
    """
    Save raw data to HDF5 format for post-processing
    
    Parameters:
    -----------
    rho, p, e_total, vr, vz, vphi : array
        Physical fields
    R, Z : array
        Coordinate grids
    step : int
        Timestep number
    """
    try:
        import h5py
        
        # Create data directory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        filename = os.path.join(data_dir, f"snapshot_{step:05d}.h5")
        
        with h5py.File(filename, 'w') as f:
            # Store fields
            f.create_dataset('density', data=rho, compression='gzip')
            f.create_dataset('pressure', data=p, compression='gzip')
            f.create_dataset('energy', data=e_total, compression='gzip')
            f.create_dataset('velocity_r', data=vr, compression='gzip')
            f.create_dataset('velocity_z', data=vz, compression='gzip')
            f.create_dataset('velocity_phi', data=vphi, compression='gzip')
            
            # Store coordinates
            f.create_dataset('R', data=R, compression='gzip')
            f.create_dataset('Z', data=Z, compression='gzip')
            
            # Store metadata
            f.attrs['step'] = step
            f.attrs['shape'] = rho.shape
            
        print(f"  → Raw data saved: {filename}")
        
    except ImportError:
        print("  WARNING: h5py not available - skipping raw data save")
        print("           Install with: pip install h5py")
    except Exception as e:
        print(f"  ERROR saving raw data: {e}")


def generate_report(rho, p, e_total, vr, vz, vphi, R, Z, step=0, 
                   save_to_file=True):
    """
    Generate text report with simulation statistics
    
    Parameters:
    -----------
    rho, p, e_total, vr, vz, vphi : array
        Physical fields
    R, Z : array
        Coordinate grids
    step : int
        Timestep number
    save_to_file : bool
        Save report to text file
    """
    # Calculate derived quantities
    T = p / (rho + 1e-20)
    cs = np.sqrt(5.0/3.0 * p / (rho + 1e-20))
    v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
    mach = v_mag / (cs + 1e-20)
    
    # Generate report text
    report = []
    report.append("="*70)
    report.append(f"SIMULATION REPORT - STEP {step}")
    report.append("="*70)
    report.append("")
    
    report.append("FIELD STATISTICS:")
    report.append("-"*70)
    report.append(f"  Density:")
    report.append(f"    Min:  {np.min(rho):.4e}    Max:  {np.max(rho):.4e}")
    report.append(f"    Mean: {np.mean(rho):.4e}    Std:  {np.std(rho):.4e}")
    report.append("")
    
    report.append(f"  Pressure:")
    report.append(f"    Min:  {np.min(p):.4e}    Max:  {np.max(p):.4e}")
    report.append(f"    Mean: {np.mean(p):.4e}    Std:  {np.std(p):.4e}")
    report.append("")
    
    report.append(f"  Temperature:")
    report.append(f"    Min:  {np.min(T):.4e}    Max:  {np.max(T):.4e}")
    report.append(f"    Mean: {np.mean(T):.4e}    Std:  {np.std(T):.4e}")
    report.append("")
    
    report.append(f"  Velocity Magnitude:")
    report.append(f"    Min:  {np.min(v_mag):.4e}    Max:  {np.max(v_mag):.4e}")
    report.append(f"    Mean: {np.mean(v_mag):.4e}    Std:  {np.std(v_mag):.4e}")
    report.append("")
    
    report.append(f"  Mach Number:")
    report.append(f"    Min:  {np.min(mach):.4e}    Max:  {np.max(mach):.4e}")
    report.append(f"    Mean: {np.mean(mach):.4e}    Std:  {np.std(mach):.4e}")
    report.append("")
    
    report.append("INTEGRATED QUANTITIES:")
    report.append("-"*70)
    
    # Calculate volumes
    dR = R[1, 0] - R[0, 0] if R.shape[0] > 1 else 1.0
    dZ = Z[0, 1] - Z[0, 0] if Z.shape[1] > 1 else 1.0
    
    total_mass = np.sum(rho * R * dR * dZ) * 2 * np.pi
    total_energy = np.sum(e_total * R * dR * dZ) * 2 * np.pi
    kinetic_energy = np.sum(0.5 * rho * v_mag**2 * R * dR * dZ) * 2 * np.pi
    internal_energy = np.sum(p / (5.0/3.0 - 1.0) * R * dR * dZ) * 2 * np.pi
    
    report.append(f"  Total Mass:       {total_mass:.4e}")
    report.append(f"  Total Energy:     {total_energy:.4e}")
    report.append(f"  Kinetic Energy:   {kinetic_energy:.4e}")
    report.append(f"  Internal Energy:  {internal_energy:.4e}")
    report.append(f"  KE/IE Ratio:      {kinetic_energy/(internal_energy+1e-20):.4f}")
    report.append("")
    
    report.append("="*70)
    
    # Print to console
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    if save_to_file:
        report_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        filename = os.path.join(report_dir, f"report_{step:05d}.txt")
        with open(filename, 'w') as f:
            f.write(report_text)
        
        print(f"\n  → Report saved: {filename}\n")


# Convenience function for complete output
def save_complete_output(rho, p, e_total, vr, vz, vphi, R, Z, step=0, 
                        save_raw=False, generate_report_file=False):
    """
    Save all outputs: plots, raw data, and reports
    
    Parameters:
    -----------
    rho, p, e_total, vr, vz, vphi : array
        Physical fields
    R, Z : array
        Coordinate grids  
    step : int
        Timestep number
    save_raw : bool
        Save raw HDF5 data
    generate_report_file : bool
        Generate text report file
    """
    # Save plots
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step)
    
    # Save raw data
    if save_raw:
        save_raw_data(rho, p, e_total, vr, vz, vphi, R, Z, step)
    
    # Generate report
    if generate_report_file:
        T = p / (rho + 1e-20)
        generate_report(rho, p, e_total, vr, vz, vphi, R, Z, step, True)
    
    # Optionally plot velocity field
    if step % 50 == 0:  # Every 50 steps
        plot_velocity_field(vr, vz, R, Z, filename=f"velocity_field_{step:05d}.png")


# Example usage function
def example_usage():
    """
    Example of how to use the enhanced save functions
    """
    print("\nExample usage of save.py:")
    print("-" * 70)
    print("Basic usage:")
    print("  from save import save_all_enhanced")
    print("  save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0)")
    print("")
    print("Complete output with raw data:")
    print("  from save import save_complete_output")
    print("  save_complete_output(rho, p, e_total, vr, vz, vphi, R, Z,")
    print("                       step=0, save_raw=True, generate_report_file=True)")
    print("")
    print("Create animations after simulation:")
    print("  from save import create_all_animations")
    print("  create_all_animations(start_step=0, end_step=1000, step_interval=10)")
    print("-" * 70)