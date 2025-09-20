
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
import matplotlib.patches as patches


# Save inside your project "figures" folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_astrophysical_contours(field, R, Z, title="Field", filename="field.png", 
                               log_scale=True, levels=20, show_black_hole=True, 
                               show_marginally_stable=True, cmap="plasma"):
    """
    Create contour plots similar to those in astrophysics papers - works with full domain data
    """
    # Clean the data
    field_clean = np.copy(field)
    
    # Handle infinities and NaNs
    mask_bad = ~np.isfinite(field_clean)
    if np.any(mask_bad):
        field_clean[mask_bad] = np.nanmin(field_clean[~mask_bad]) if np.any(~mask_bad) else 1e-20
        print(f"Warning: {np.sum(mask_bad)} bad values in {title}")
    
    # Ensure positive values for log scale
    if log_scale:
        field_clean = np.maximum(field_clean, 1e-20)
    
    # Statistics
    field_min, field_max = np.min(field_clean), np.max(field_clean)
    field_mean = np.mean(field_clean)
    
    print(f"{title} stats: min={field_min:.2e}, max={field_max:.2e}, mean={field_mean:.2e}")
    
    # --- Create the plot using existing full domain data ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # Square plot for full view
    
    # Create contour levels
    if log_scale and field_min > 0:
        # Logarithmic levels
        log_min = np.log10(field_min)
        log_max = np.log10(field_max)
       # if log_max - log_min > 6:  # If range is too large, limit it
        #    log_min = log_max - 6
        contour_levels = np.logspace(log_min, log_max, levels)
        norm = LogNorm(vmin=field_min, vmax=field_max)
    else:
        # Linear levels
        contour_levels = np.linspace(field_min, field_max, levels)
        norm = None
    
    # Main contour plot (filled) - use existing full domain data
    if norm:
        cs_filled = ax.contourf(R, Z, field_clean, levels=contour_levels, 
                               cmap=cmap, norm=norm, extend='both')
    else:
        cs_filled = ax.contourf(R, Z, field_clean, levels=contour_levels, 
                               cmap=cmap, extend='both')
    
    # Contour lines (like in the paper)
    cs_lines = ax.contour(R, Z, field_clean, levels=contour_levels[::2], 
                         colors='black', alpha=0.4, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(cs_filled, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(title, rotation=270, labelpad=20, fontsize=12)
    
    # Mark important astrophysical features
    if show_black_hole:
        # Black hole at origin (r=1 in Schwarzschild units)
        bh_circle = patches.Circle((0, 0), 1.0, color='black', fill=True, 
                                  zorder=10, label='Black Hole')
        ax.add_patch(bh_circle)
    
    if show_marginally_stable:
        # Marginally stable orbit at r=3
        ms_circle = patches.Circle((0, 0), 3.0, color='white', fill=False, 
                                  linestyle='--', linewidth=2, alpha=0.8, 
                                  zorder=9, label='Marginally Stable Orbit')
        ax.add_patch(ms_circle)
    
    # Add equatorial plane and rotation axis lines
    r_max_plot = np.max(np.abs(R))
    z_max_plot = np.max(np.abs(Z))
    ax.axhline(y=0, color='white', linestyle=':', alpha=0.5, linewidth=1, label='Equatorial Plane')
    ax.axvline(x=0, color='white', linestyle=':', alpha=0.5, linewidth=1, label='Rotation Axis')
    
    # Formatting similar to research papers
    ax.set_xlabel('R (Schwarzschild radii)', fontsize=12)
    ax.set_ylabel('Z (Schwarzschild radii)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_aspect('equal')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Use the actual domain limits from your simulation
    ax.set_xlim(np.min(R), np.max(R))
    ax.set_ylim(np.min(Z), np.max(Z))
    
    # Add legend if astrophysical features are shown
    if show_black_hole or show_marginally_stable:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save CSV data
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    r_vals = R[:, 0] if R.shape[1] > 1 else R.flatten()
    z_vals = Z[0, :] if Z.shape[0] > 1 else Z.flatten()
    
    with open(csv_path, "w") as f:
        f.write("r/z," + ",".join(map(str, z_vals)) + "\n")
        for i, r_val in enumerate(r_vals):
            if i < field_clean.shape[0]:
                row = [str(r_val)] + [str(field_clean[i, j]) for j in range(min(len(z_vals), field_clean.shape[1]))]
                f.write(",".join(row) + "\n")
    
    # Save CSV data
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    r_vals = R[:, 0] if R.shape[1] > 1 else R.flatten()
    z_vals = Z[0, :] if Z.shape[0] > 1 else Z.flatten()
    
    with open(csv_path, "w") as f:
        f.write("r/z," + ",".join(map(str, z_vals)) + "\n")
        for i, r_val in enumerate(r_vals):
            if i < field_clean.shape[0]:
                row = [str(r_val)] + [str(field_clean[i, j]) for j in range(min(len(z_vals), field_clean.shape[1]))]
                f.write(",".join(row) + "\n")

def plot_field_enhanced(field, R, Z, title="Field", cmap="inferno", filename="field.png", 
                       log_scale=False, vmin=None, vmax=None, show_contours=False):
    """
    Enhanced plotting with better handling of extreme values and debugging info
    """
    # Use the new astrophysical contour function
    plot_astrophysical_contours(field, R, Z, title=title, filename=filename, 
                               log_scale=log_scale, cmap=cmap)

def plot_comparative_contours(fields_dict, R, Z, filename="comparative_fields.png"):
    """
    Create a multi-panel figure comparing different fields - works with existing full domain data
    """
    n_fields = len(fields_dict)
    if n_fields == 0:
        return
    
    # Determine subplot layout
    if n_fields <= 2:
        rows, cols = 1, n_fields
        figsize = (10 * n_fields, 8)
    elif n_fields <= 4:
        rows, cols = 2, 2
        figsize = (16, 16)
    elif n_fields <= 6:
        rows, cols = 2, 3
        figsize = (24, 16)
    else:
        rows, cols = 3, 3
        figsize = (24, 24)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_fields == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Color maps for different physical quantities
    cmaps = {
        'density': 'plasma',
        'pressure': 'viridis', 
        'temperature': 'hot',
        'energy': 'inferno',
        'velocity': 'RdBu_r',
        'mach': 'coolwarm'
    }
    
    for idx, (field_name, field_data) in enumerate(fields_dict.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Clean data
        field_clean = np.copy(field_data)
        field_clean[~np.isfinite(field_clean)] = np.nanmin(field_clean[np.isfinite(field_clean)])
        
        # Choose appropriate colormap
        cmap = 'plasma'  # default
        for key, color in cmaps.items():
            if key.lower() in field_name.lower():
                cmap = color
                break
        
        # Determine if log scale is appropriate
        log_scale = (np.min(field_clean) > 0 and 
                    np.max(field_clean) / np.min(field_clean) > 100)
        
        if log_scale:
            field_clean = np.maximum(field_clean, 1e-20)
            norm = LogNorm(vmin=np.min(field_clean), vmax=np.max(field_clean))
        else:
            norm = None
        
        # Create contour plot using existing full domain data
        levels = 15
        if log_scale:
            log_min, log_max = np.log10(np.min(field_clean)), np.log10(np.max(field_clean))
            contour_levels = np.logspace(log_min, log_max, levels)
        else:
            contour_levels = np.linspace(np.min(field_clean), np.max(field_clean), levels)
        
        cs = ax.contourf(R, Z, field_clean, levels=contour_levels, 
                        cmap=cmap, norm=norm, extend='both')
        ax.contour(R, Z, field_clean, levels=contour_levels[::2], 
                  colors='black', alpha=0.3, linewidths=0.5)
        
        # Add black hole and marginally stable orbit
        bh_circle = patches.Circle((0, 0), 1.0, color='black', fill=True, zorder=10)
        ms_circle = patches.Circle((0, 0), 3.0, color='white', fill=False, 
                                  linestyle='--', linewidth=1.5, alpha=0.7, zorder=9)
        ax.add_patch(bh_circle)
        ax.add_patch(ms_circle)
        
        # Add reference lines
        ax.axhline(y=0, color='white', linestyle=':', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='white', linestyle=':', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('R (r_g)')
        ax.set_ylabel('Z (r_g)')
        ax.set_title(field_name)
        ax.set_aspect('equal')
        
        # Use actual domain limits from your simulation
        ax.set_xlim(np.min(R), np.max(R))
        ax.set_ylim(np.min(Z), np.max(Z))
        
        # Add colorbar
        cbar = plt.colorbar(cs, ax=ax, shrink=0.6)
        if log_scale:
            cbar.set_label(f'log({field_name})', rotation=270, labelpad=15)
        else:
            cbar.set_label(field_name, rotation=270, labelpad=15)
    
    # Hide unused subplots
    for idx in range(len(fields_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()



def save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0):
    """
    Enhanced saving with astrophysical contour plots similar to research papers
    """
    print(f"\n--- Saving astrophysical data for step {step} ---")
    
    # --- Basic thermodynamic quantities (like Figures 2-4 in paper) ---
    plot_astrophysical_contours(rho, R, Z, title="Density", 
                               filename=f"density_{step:05d}.png", 
                               log_scale=True, cmap="plasma")
    
    plot_astrophysical_contours(p, R, Z, title="Pressure", 
                               filename=f"pressure_{step:05d}.png", 
                               log_scale=True, cmap="viridis")
    
    plot_astrophysical_contours(e_total, R, Z, title="Total Energy", 
                               filename=f"energy_{step:05d}.png", 
                               log_scale=True, cmap="inferno")
    
    # --- Temperature (like Figure 2b, 4, 7b, 9 in paper) ---
    T = p / (rho + 1e-20)  # Temperature in code units
    plot_astrophysical_contours(T, R, Z, title="Temperature (keV)", 
                               filename=f"temperature_{step:05d}.png", 
                               log_scale=True, cmap="hot")
    
    # --- Velocity components ---
    plot_astrophysical_contours(vr, R, Z, title="Radial Velocity", 
                               filename=f"vr_{step:05d}.png", 
                               log_scale=False, cmap="RdBu_r")
    
    plot_astrophysical_contours(vz, R, Z, title="Vertical Velocity", 
                               filename=f"vz_{step:05d}.png", 
                               log_scale=False, cmap="RdBu_r")
    v_mag = np.sqrt(vr**2 + vz**2 + vphi**2)
    plot_astrophysical_contours(v_mag, R, Z, title="Vertical Velocity", 
                               filename=f"vmag_{step:05d}.png", 
                               log_scale=False, cmap="RdBu_r")
    
    
    if np.any(np.abs(vphi) > 1e-10):
        plot_astrophysical_contours(vphi, R, Z, title="Azimuthal Velocity", 
                                   filename=f"vphi_{step:05d}.png", 
                                   log_scale=False, cmap="RdBu_r")
    
    # --- Vector field visualization ---
   
    
    # --- Derived astrophysical quantities ---
    cs = np.sqrt(np.maximum(5.0/3.0 * p / (rho + 1e-20), 1e-20))  # Sound speed
    mach_r = np.abs(vr) / cs  # Radial Mach number
    
    plot_astrophysical_contours(cs, R, Z, title="Sound Speed", 
                               filename=f"sound_speed_{step:05d}.png", 
                               log_scale=True, cmap="cool")
    
    plot_astrophysical_contours(mach_r, R, Z, title="Radial Mach Number", 
                               filename=f"mach_{step:05d}.png", 
                               log_scale=False, cmap="coolwarm")
    
    # --- Specific angular momentum ---
    l_spec = R * vphi
    if np.any(np.abs(l_spec) > 1e-10):
        plot_astrophysical_contours(l_spec, R, Z, title="Specific Angular Momentum", 
                                   filename=f"angular_momentum_{step:05d}.png", 
                                   log_scale=False, cmap="RdBu_r")
    
    # --- Multi-panel comparison (like in research papers) ---
    if step % 50 == 0:  # Every 50th step
        comparison_fields = {
            'Density': rho,
            'Temperature': T,
            'Pressure': p,
            'Radial Velocity': vr,
            'Mach Number': mach_r,
            'Sound Speed': cs
        }
        plot_comparative_contours(comparison_fields, R, Z, 
                                filename=f"comparison_{step:05d}.png")
    
    # Summary statistics
    print(f"Astrophysical data summary for step {step}:")
    print(f"  Density range: [{np.min(rho):.2e}, {np.max(rho):.2e}]")
    print(f"  Temperature range: [{np.min(T):.2e}, {np.max(T):.2e}] keV")
    print(f"  Pressure range: [{np.min(p):.2e}, {np.max(p):.2e}]")
    print(f"  Max radial velocity: {np.max(np.abs(vr)):.3f} c")
    print(f"  Max Mach number: {np.max(mach_r):.3f}")

def save_all(rho, p, e_total, R, Z, step=0):
    """
    Backward compatibility function - calls enhanced astrophysical version
    """
    # Create dummy velocity components if not provided
    vphi = np.zeros_like(rho)
    vr = np.zeros_like(rho)
    vz = np.zeros_like(rho)
    
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step)

def plot_radial_profiles(fields_dict, R, Z, radii=[5, 10, 20, 30], filename="radial_profiles.png"):
    """
    Plot radial profiles at different radii (similar to paper analysis)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Spherical radius grid
    Rg = np.sqrt(R**2 + Z**2)
    
    # Colors for different radii
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    plot_idx = 0
    for field_name, field_data in fields_dict.items():
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        
        for i, radius in enumerate(radii):
            if radius > np.max(Rg):
                continue
                
            # Find points near this spherical radius
            mask_radius = (np.abs(Rg - radius) < 1.0) & (Z >= 0)  # Upper half plane
            
            if np.any(mask_radius):
                # Extract angle and field values
                theta_vals = np.arctan2(Z[mask_radius], R[mask_radius])
                field_vals = field_data[mask_radius]
                
                # Sort by angle
                sort_idx = np.argsort(theta_vals)
                theta_sorted = theta_vals[sort_idx]
                field_sorted = field_vals[sort_idx]
                
                ax.semilogy(theta_sorted * 180/np.pi, np.abs(field_sorted), 
                           'o-', color=colors[i % len(colors)], 
                           label=f'r = {radius} r_g', markersize=3, linewidth=1.5)
        
        ax.set_xlabel("Polar Angle (degrees)")
        ax.set_ylabel(f"|{field_name}|")
        ax.set_title(f"{field_name} vs Angle")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_equatorial_profile(fields_dict, R, Z, filename="equatorial_profile.png"):
    """
    Plot profiles along the equatorial plane (z=0) - important for accretion disk physics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Find equatorial plane (z closest to 0)
    z_idx = np.argmin(np.abs(Z[0, :]))
    r_vals = R[:, z_idx]
    
    plot_idx = 0
    for field_name, field_data in fields_dict.items():
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        field_vals = field_data[:, z_idx]
        
        ax.loglog(r_vals, np.abs(field_vals), 'o-', linewidth=2, markersize=4)
        ax.set_xlabel("Radius (r_g)")
        ax.set_ylabel(f"|{field_name}|")
        ax.set_title(f"{field_name} along Equatorial Plane")
        ax.grid(True, alpha=0.3, which='both')
        
        # Mark important radii
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='r_ms = 3')
        ax.axvline(x=6, color='orange', linestyle=':', alpha=0.7, label='r = 6')
        ax.legend(fontsize=10)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

# Keep all other functions unchanged
def create_animation(field_name="density", start_step=0, end_step=100, step_interval=10):
    """
    Create animation from saved images
    """
    try:
        import imageio
        import glob
        
        # Find matching files
        pattern = os.path.join(OUTPUT_DIR, f"{field_name}_*.png")
        files = sorted(glob.glob(pattern))
        
        if len(files) < 2:
            print(f"Not enough files found for {field_name} animation")
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
            print(f"Not enough files in range for {field_name} animation")
            return
        
        # Create animation
        images = []
        for filename in filtered_files:
            images.append(imageio.imread(filename))
        
        output_file = os.path.join(OUTPUT_DIR, f"{field_name}_animation.gif")
        imageio.mimsave(output_file, images, duration=0.5)
        print(f"Animation saved: {output_file}")
        
    except ImportError:
        print("imageio not available - cannot create animations")
    except Exception as e:
        print(f"Error creating animation: {e}")

def cleanup_old_files(max_files=200):
    """
    Clean up old files to prevent disk space issues
    """
    try:
        import glob
        
        # Get all PNG files
        files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
        
        if len(files) > max_files:
            # Sort by modification time
            files.sort(key=os.path.getmtime)
            
            # Remove oldest files
            files_to_remove = files[:len(files) - max_files]
            for f in files_to_remove:
                os.remove(f)
            
            print(f"Cleaned up {len(files_to_remove)} old files")
    
    except Exception as e:
        print(f"Error during cleanup: {e}")