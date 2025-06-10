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
    # Data is converted to float32 for precision in flux calculations
    with fits.open(image) as hdul:
        data = hdul[0].data.astype('f4')
        header = hdul[0].header  # (optional: may be useful for tracking metadata)

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
        pos = tuple(row['position'])  # Ensure the position is hashable as a dict key
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
            time = header.get("JD", header.get("DATE-OBS", "unknown"))
        print(f"{fname}: {time}")

        phot = do_aperture_photometry(
            image=fname,
            positions=[position],
            radii=[aperture_radius],
            sky_radius_in=sky_radius_in,
            sky_annulus_width=sky_annulus_width,
        )
        flux = phot[0]['net_flux']
        times.append(time)
        fluxes.append(flux)

    table = Table([times, fluxes], names=("time", "flux"))
    csv_path = os.path.join(directory, f"{output_basename}.csv")
    png_path = os.path.join(directory, f"{output_basename}.png")
    table.write(csv_path, format="ascii.csv", overwrite=True)

    plt.figure()
    plt.plot(times, fluxes, "ko-")
    plt.xlabel("Time")
    plt.ylabel("Net Flux")
    plt.title(f"Light Curve: {output_basename}")
    plt.grid(True)
    plt.savefig(png_path)
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