# save.py - Enhanced with better visualization and debugging
import matplotlib.pyplot as plt
import numpy as np
import os

# Save inside your project "figures" folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_field_enhanced(field, R, Z, title="Field", cmap="inferno", filename="field.png", 
                       log_scale=False, vmin=None, vmax=None, show_contours=False):
    """
    Enhanced plotting with better handling of extreme values and debugging info
    """
    # --- Data analysis and cleaning ---
    field_clean = np.copy(field)
    
    # Handle infinities and NaNs
    mask_bad = ~np.isfinite(field_clean)
    if np.any(mask_bad):
        field_clean[mask_bad] = np.nanmin(field_clean[~mask_bad]) if np.any(~mask_bad) else 0.0
        print(f"Warning: {np.sum(mask_bad)} bad values in {title}")
    
    # Statistics
    field_min, field_max = np.min(field_clean), np.max(field_clean)
    field_mean, field_std = np.mean(field_clean), np.std(field_clean)
    
    print(f"{title} stats: min={field_min:.2e}, max={field_max:.2e}, mean={field_mean:.2e}, std={field_std:.2e}")
    
    # --- Plotting with better ranges ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Linear scale with clipped range
    if vmin is None or vmax is None:
        # Use percentile-based clipping to avoid extreme outliers
        p1, p99 = np.percentile(field_clean, [1, 99])
        if vmin is None:
            vmin = max(p1, field_min) if field_min > 0 else p1
        if vmax is None:
            vmax = min(p99, field_max) if p99 < field_max * 0.1 else field_max
    
    # Clip field for visualization
    field_plot = np.clip(field_clean, vmin, vmax)
    
    im1 = ax1.pcolormesh(R, Z, field_plot, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_xlabel("r")
    ax1.set_ylabel("z")
    ax1.set_title(f"{title} (Linear, clipped)")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    if show_contours:
        contour_levels = np.linspace(vmin, vmax, 10)
        ax1.contour(R, Z, field_plot, levels=contour_levels, colors='white', alpha=0.3, linewidths=0.5)
    
    # Right plot: Log scale (if requested and field is positive)
    if log_scale and np.all(field_clean > 0):
        field_log = np.log10(np.maximum(field_clean, 1e-20))
        im2 = ax2.pcolormesh(R, Z, field_log, shading="auto", cmap=cmap)
        ax2.set_title(f"{title} (Log₁₀)")
        plt.colorbar(im2, ax=ax2, label=f"log₁₀({title})")
    else:
        # Show difference from mean for better structure visualization
        field_diff = field_clean - field_mean
        im2 = ax2.pcolormesh(R, Z, field_diff, shading="auto", cmap="RdBu_r")
        ax2.set_title(f"{title} (Deviation from mean)")
        plt.colorbar(im2, ax=ax2, label=f"Δ{title}")
    
    ax2.set_xlabel("r")
    ax2.set_ylabel("z")
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

    # --- CSV saving (same as before) ---
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    r_vals = R[:, 0]  # First column of R
    z_vals = Z[0, :]  # First row of Z

    with open(csv_path, "w") as f:
        # Write header
        f.write("r/z," + ",".join(map(str, z_vals)) + "\n")
        # Write data rows
        for i, r_val in enumerate(r_vals):
            row = [str(r_val)] + [str(field[i, j]) for j in range(len(z_vals))]
            f.write(",".join(row) + "\n")

def plot_vector_field(vr, vz, R, Z, title="Velocity Field", filename="velocity_field.png", 
                     subsample=4, scale=1.0):
    """
    Plot vector field with streamlines
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vector magnitude
    v_mag = np.sqrt(vr**2 + vz**2)
    
    # Left: Quiver plot (subsampled for clarity)
    R_sub = R[::subsample, ::subsample]
    Z_sub = Z[::subsample, ::subsample]
    vr_sub = vr[::subsample, ::subsample]
    vz_sub = vz[::subsample, ::subsample]
    v_mag_sub = v_mag[::subsample, ::subsample]
    
    im1 = ax1.pcolormesh(R, Z, v_mag, shading="auto", cmap="plasma", alpha=0.7)
    q = ax1.quiver(R_sub, Z_sub, vr_sub, vz_sub, v_mag_sub, 
                   scale=scale, cmap="plasma", alpha=0.8)
    ax1.set_xlabel("r")
    ax1.set_ylabel("z")
    ax1.set_title(f"{title} (Vectors)")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label="Velocity Magnitude")
    
    # Right: Streamlines
    try:
        ax2.streamplot(R, Z, vr, vz, density=1.5, color=v_mag, cmap="plasma", 
                      linewidth=1.5, arrowsize=1.2)
        im2 = ax2.pcolormesh(R, Z, v_mag, shading="auto", cmap="plasma", alpha=0.3)
        ax2.set_xlabel("r")
        ax2.set_ylabel("z")
        ax2.set_title(f"{title} (Streamlines)")
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label="Velocity Magnitude")
    except:
        # Fallback if streamplot fails
        im2 = ax2.pcolormesh(R, Z, v_mag, shading="auto", cmap="plasma")
        ax2.set_xlabel("r")
        ax2.set_ylabel("z")
        ax2.set_title(f"{title} (Magnitude)")
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label="Velocity Magnitude")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_radial_profiles(fields_dict, R, Z, radii=[5, 10, 20, 30], filename="radial_profiles.png"):
    """
    Plot radial profiles of various fields at different radii
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
                
                ax.plot(theta_sorted * 180/np.pi, field_sorted, 
                       'o-', color=colors[i % len(colors)], 
                       label=f'r = {radius}', markersize=3, linewidth=1.5)
        
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel(field_name)
        ax.set_title(f"{field_name} vs Angle")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_equatorial_profile(fields_dict,R, Z, filename="equatorial_profile.png"):
    """
    Plot profiles along the equatorial plane (z=0)
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
        
        ax.semilogy(r_vals, np.abs(field_vals), 'o-', linewidth=2, markersize=4)
        ax.set_xlabel("Radius r")
        ax.set_ylabel(f"|{field_name}|")
        ax.set_title(f"{field_name} along Equator")
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step=0):
    """
    Enhanced saving with more fields and better visualization
    """
    print(f"\n--- Saving data for step {step} ---")
    
    # Basic thermodynamic quantities
    plot_field_enhanced(rho, R, Z, title="Density", filename=f"density_{step:05d}.png", 
                       log_scale=True, show_contours=True)
    
    plot_field_enhanced(p, R, Z, title="Pressure", filename=f"pressure_{step:05d}.png", 
                       log_scale=True, show_contours=True)
    
    plot_field_enhanced(e_total, R, Z, title="Total Energy", filename=f"energy_{step:05d}.png", 
                       log_scale=True)
    
    # Velocity components
    plot_field_enhanced(vr, R, Z, title="Radial Velocity", cmap="RdBu_r", 
                       filename=f"vr_{step:05d}.png")
    
    plot_field_enhanced(vz, R, Z, title="Vertical Velocity", cmap="RdBu_r", 
                       filename=f"vz_{step:05d}.png")
    
    if np.any(np.abs(vphi) > 1e-10):
        plot_field_enhanced(vphi, R, Z, title="Azimuthal Velocity", cmap="RdBu_r", 
                           filename=f"vphi_{step:05d}.png")
    
    # Vector field visualization
    plot_vector_field(vr, vz, R, Z, title="Velocity Field", 
                     filename=f"velocity_field_{step:05d}.png", subsample=6)
    
    # Derived quantities
    cs = np.sqrt(np.maximum(5.0/3.0 * p / (rho + 1e-20), 1e-20))  # Sound speed
    mach_r = np.abs(vr) / cs  # Radial Mach number
    v_total = np.sqrt(vr**2 + vz**2 + vphi**2)  # Total velocity magnitude
    
    plot_field_enhanced(cs, R, Z, title="Sound Speed", filename=f"cs_{step:05d}.png", 
                       log_scale=True)
    
    plot_field_enhanced(mach_r, R, Z, title="Radial Mach Number", filename=f"mach_{step:05d}.png")
    
    plot_field_enhanced(v_total, R, Z, title="Velocity Magnitude", filename=f"vmag_{step:05d}.png")
    
    # Temperature (assuming ideal gas)
    T = p / (rho + 1e-20)  # Temperature in code units
    plot_field_enhanced(T, R, Z, title="Temperature", filename=f"temperature_{step:05d}.png", 
                       log_scale=True)
    
    # Specific angular momentum
    l_spec = R * vphi  # Specific angular momentum
    if np.any(np.abs(l_spec) > 1e-10):
        plot_field_enhanced(l_spec, R, Z, title="Specific Angular Momentum", 
                           cmap="RdBu_r", filename=f"angular_momentum_{step:05d}.png")
    
    # Entropy (assuming ideal gas: s ∝ ln(p/ρ^γ))
    gamma = 5.0/3.0
    entropy = np.log(p / (rho**gamma + 1e-20))
    plot_field_enhanced(entropy, R, Z, title="Entropy", cmap="viridis", 
                       filename=f"entropy_{step:05d}.png")
    
    # Vorticity (curl of velocity in 2D: ∂vz/∂r - ∂vr/∂z)
    if R.shape[0] > 2 and R.shape[1] > 2:
        vorticity = np.zeros_like(vr)
        dr = R[1, 0] - R[0, 0]
        dz = Z[0, 1] - Z[0, 0]
        
        # Interior points
        vorticity[1:-1, 1:-1] = ((vz[2:, 1:-1] - vz[:-2, 1:-1]) / (2 * dr) - 
                                (vr[1:-1, 2:] - vr[1:-1, :-2]) / (2 * dz))
        
        if np.any(np.abs(vorticity) > 1e-10):
            plot_field_enhanced(vorticity, R, Z, title="Vorticity", cmap="RdBu_r", 
                               filename=f"vorticity_{step:05d}.png")
    
    # Velocity divergence (useful for identifying shocks)
    div_v = np.zeros_like(vr)
    dr = R[1, 0] - R[0, 0] if R.shape[0] > 1 else 1.0
    dz = Z[0, 1] - Z[0, 0] if R.shape[1] > 1 else 1.0
    
    for i in range(1, R.shape[0]-1):
        for j in range(1, R.shape[1]-1):
            r_val = R[i, j]
            if r_val > 1e-10:
                # ∇·v = (1/r)∂(rvr)/∂r + ∂vz/∂z
                dr_term = ((R[i+1, j] * vr[i+1, j] - R[i-1, j] * vr[i-1, j]) / 
                          (2 * dr * r_val))
                dz_term = (vz[i, j+1] - vz[i, j-1]) / (2 * dz)
                div_v[i, j] = dr_term + dz_term
    
    if np.any(np.abs(div_v) > 1e-10):
        plot_field_enhanced(div_v, R, Z, title="Velocity Divergence", cmap="RdBu_r", 
                           filename=f"divergence_{step:05d}.png")
    
    # Radial profiles (only for significant steps)
    if step % 50 == 0:  # Every 50th step
        fields_for_profile = {
            'Density': rho,
            'Pressure': p,
            'Radial Velocity': vr,
            'Mach Number': mach_r
        }
        plot_radial_profiles(fields_for_profile, R, Z, 
                           filename=f"radial_profiles_{step:05d}.png")
        
        plot_equatorial_profile(fields_for_profile, R, Z, 
                              filename=f"equatorial_profile_{step:05d}.png")
    
    # Summary statistics
    print(f"Data summary for step {step}:")
    print(f"  Density: [{np.min(rho):.2e}, {np.max(rho):.2e}]")
    print(f"  Pressure: [{np.min(p):.2e}, {np.max(p):.2e}]")
    print(f"  |v_r| max: {np.max(np.abs(vr)):.3f}")
    print(f"  |v_z| max: {np.max(np.abs(vz)):.3f}")
    print(f"  Mach max: {np.max(mach_r):.3f}")
    print(f"  Temperature range: [{np.min(T):.2e}, {np.max(T):.2e}]")

def save_all(rho, p, e_total, R, Z, step=0):
    """
    Backward compatibility function - calls enhanced version
    """
    # Create dummy vphi if not provided (for backward compatibility)
    vphi = np.zeros_like(rho)
    vr = np.zeros_like(rho)  # Will need to be computed from momentum if available
    vz = np.zeros_like(rho)
    
    save_all_enhanced(rho, p, e_total, vr, vz, vphi, R, Z, step)

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