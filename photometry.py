def do_aperture_photometry(
    image,
    positions,
    radii,
    sky_radius_in,
    sky_annulus_width,
):
    """
    Perform circular aperture photometry on one or more target positions in an image,
    including sky background subtraction using an annulus.

    Parameters
    ----------
    image : str
        Path to the FITS file containing the science data.
    positions : list of tuples
        List of (x, y) coordinates of the target(s) to measure.
    radii : list of float
        Radii (in pixels) to use for the circular aperture(s).
    sky_radius_in : float
        Inner radius of the sky background annulus (in pixels).
    sky_annulus_width : float
        Width of the sky annulus (in pixels).

    Returns
    -------
    final_table : astropy.table.QTable
        Combined photometry table with aperture sums, background estimates,
        net flux, and metadata for each aperture radius at each position.
    """
    import numpy as np
    from astropy.io import fits
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.table import vstack

    # Step 1: Open the FITS file and extract image data
    with fits.open(image) as hdul:
        data = hdul[0].data.astype('f4')
        header = hdul[0].header 

    results = []

    # Step 2: Loop over all requested target positions and aperture radii
    for pos in positions:
        for r in radii:
            # Step 2a: Define circular aperture for the target and annular aperture for sky
            aperture = CircularAperture(pos, r=r)
            annulus = CircularAnnulus(pos, r_in=sky_radius_in, r_out=sky_radius_in + sky_annulus_width)

            # Step 2b: Perform aperture photometry on both apertures simultaneously
            apers = [aperture, annulus]
            phot_table = aperture_photometry(data, apers)

            # Step 2c: Estimate mean sky background from annulus
            annulus_area = annulus.area
            bkg_mean = phot_table['aperture_sum_1'] / annulus_area  # mean background per pixel
            bkg_total = bkg_mean * aperture.area                    # total background in aperture

            # Step 2d: Subtract background and store relevant info
            net_flux = phot_table['aperture_sum_0'] - bkg_total
            phot_table['net_flux'] = net_flux
            phot_table['position'] = [pos]
            phot_table['radius'] = r

            results.append(phot_table)

    # Step 3: Combine results from all positions and radii into one table
    final_table = vstack(results)
    return final_table

    #Combine
    final_table = vstack(results)
    return final_table


def plot_radial_profile(aperture_photometry_data, output_filename="radial_profile.png"):
    """
    Plot a radial flux profile from aperture photometry measurements.

    Parameters
    ----------
    aperture_photometry_data : astropy.table.QTable
        Table output from do_aperture_photometry(). Must contain 'radius', 'net_flux',
        and 'position' columns.
    output_filename : str
        Path to save the radial profile plot as a PNG.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # Step 1: Group the data by position — this supports multiple targets
    profile_data = {}
    for row in aperture_photometry_data:
        pos = tuple(row['position']) 
        r = row['radius']
        flux = row['net_flux']

        if pos not in profile_data:
            profile_data[pos] = []
        profile_data[pos].append((r, flux))

    # Step 2: Create a new plot
    plt.figure(figsize=(8, 6))

    # Step 3: For each target position, plot the net flux as a function of radius
    for pos, profile in profile_data.items():
        profile = sorted(profile)  # Sort by increasing aperture radius
        radii, fluxes = zip(*profile)
        plt.plot(radii, fluxes, marker='o', label=f'Position {pos}')

    # Step 4: Add a vertical dashed line for the start of the sky annulus
    # Note: Assumes sky_radius_in + width = 3 pixels offset from last photometry radius
    sky_radius = aperture_photometry_data[0]['radius'] + 3
    plt.axvline(sky_radius, color='gray', linestyle='--', label='Sky annulus start')

    # Step 5: Format the plot for clarity
    plt.xlabel("Aperture Radius (pixels)")
    plt.ylabel("Net Flux (e⁻)")
    plt.title("Radial Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()


def generate_light_curve(
    directory,
    position,
    aperture_radius=5,
    sky_radius_in=8,
    sky_annulus_width=4,
    output_basename="lightcurve"
):
    """
    Generate a light curve by performing aperture photometry on all FITS files in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing reduced FITS images.
    position : tuple
        (x, y) pixel coordinates of the target star.
    aperture_radius : float
        Radius of the aperture used for photometry.
    sky_radius_in : float
        Inner radius of the sky annulus.
    sky_annulus_width : float
        Width of the sky annulus.
    output_basename : str
        Base filename for output CSV and PNG files.

    Returns
    -------
    None
    """
    import os
    import matplotlib.pyplot as plt
    from astropy.table import Table

    fits_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".fits")])
    times = []
    fluxes = []

    for fname in fits_files:
        from astropy.io import fits
        with fits.open(fname) as hdul:
            header = hdul[0].header
            jd = header["JD"]
            time_val = float(jd)

        phot = do_aperture_photometry(
            image=fname,
            positions=[position],
            radii=[aperture_radius],
            sky_radius_in=sky_radius_in,
            sky_annulus_width=sky_annulus_width,
        )
        flux = phot[0]['net_flux']
        times.append(time_val)
        fluxes.append(flux)

    table = Table([times, fluxes], names=("time", "flux"))
    csv_path = os.path.join(directory, f"{output_basename}.csv")
    png_path = os.path.join(directory, f"{output_basename}.png")
    table.write(csv_path, format="ascii.csv", overwrite=True)

    from astropy.time import Time
    import numpy as np

    fluxes = np.array(fluxes)

    # Store unnormalized flux for error calculation
    raw_fluxes = fluxes.copy()

    # Estimate simple error bars using non-negative flux for sqrt
    aperture_area = np.pi * aperture_radius**2
    sky_std = 20  # assumed RMS background in e-/pix
    flux_errors = np.sqrt(np.maximum(raw_fluxes, 0) + aperture_area * sky_std**2)

    # Normalize both flux and error
    median_flux = np.median(raw_fluxes)
    fluxes /= median_flux
    flux_errors /= median_flux

    times_jd = [float(t) for t in times]
    times_rel = np.array(times_jd) - np.min(times_jd)  # Relative JD time axis

    plt.figure(figsize=(10, 6))
    plt.errorbar(times_rel, fluxes, yerr=flux_errors, fmt="ko-", markersize=4, linewidth=1, capsize=2)

    # Choose expected mid-transit time based on target
    mid_transit_jd = None
    if "TOI-1199" in directory:
        mid_transit_jd = 2460826.70764
    elif "GJ-486" in directory:
        mid_transit_jd = 2460826.67986
        
    # Only plot if known and within ±1.5 days of observed data
    if mid_transit_jd:
        mid_transit_rel = mid_transit_jd - np.min(times_jd)
        # Show the line if within ±1.5 days of the observed data
        if -1.5 <= mid_transit_rel <= (np.max(times_jd) - np.min(times_jd)) + 1.5:
            plt.axvline(mid_transit_rel, color='blue', linestyle='--', label='Expected Mid-Transit')

    plt.xlabel("Relative Julian Date", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.title("TOI-1199 Transit Light Curve (ARCSAT, r-band)", fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Disable Julian Date offset notation for better readability
    from matplotlib.ticker import ScalarFormatter
    ax = plt.gca()
    formatter = ScalarFormatter()
    formatter.set_useOffset(False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    plt.draw()

    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(png_path.replace(".png", ".pdf"))
    plt.close()
if __name__ == "__main__":
    # Known target positions
    TOI_1199_pos = (510, 507)
    GJ_486_pos = (520, 509)

    # Generate light curve for GJ-486
    generate_light_curve(
        directory="reductions/GJ-486",
        position=GJ_486_pos,
        output_basename="gj486_lightcurve"
    )

    # Generate light curve for TOI-1199 with improved photometry parameters
    generate_light_curve(
        directory="reductions/TOI-1199",
        position=TOI_1199_pos,
        aperture_radius=7,
        sky_radius_in=12,
        sky_annulus_width=6,
        output_basename="toi1199_lightcurve"
    )

    print("Photometry complete. Light curves saved to each reductions/ subfolder.")
