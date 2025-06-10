

# Global imports
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import os



def create_median_bias(bias_list, median_bias_filename):
    """
    Create a median bias frame using sigma clipping and save it to a FITS file.

    Parameters
    ----------
    bias_list : list of str
        A list of file paths to individual bias frames (FITS format).
    median_bias_filename : str
        The name of the output FITS file where the median bias frame will be saved.

    Returns
    -------
    median_bias_filled : ndarray
        A 2D numpy array representing the final, sigma-clipped median bias frame.
    """

    # Step 1: Load all bias images into a list
    # Each bias frame is read from disk and converted to 32-bit float to standardize data type.
    # These frames are stored in a list to later be combined into a 3D stack for pixel-wise operations.
    bias_data_list = []
    for bias_file in bias_list:
        with fits.open(bias_file) as hdul:
            data = hdul[0].data.astype('f4')  # Convert to 32-bit float
            bias_data_list.append(data)

    # Step 2: Stack all 2D bias arrays into a 3D array
    # This 3D stack has shape (N, height, width), where N is the number of bias frames.
    # Allows for vectorized pixel-by-pixel operations across all frames.
    data_stack = np.stack(bias_data_list, axis=0)

    # Step 3: Apply sigma clipping along the stack axis (axis=0)
    # Sigma clipping removes statistical outliers (e.g., cosmic rays, read noise spikes)
    # at each pixel by masking values beyond 3 standard deviations from the median.
    # The resulting masked array retains the core signal across the set.
    bias_clipped = sigma_clip(data_stack, cenfunc='median', sigma=3, axis=0)

    # Step 4: Compute the median across the clipped stack
    # This produces a 2D frame representing the typical value for each pixel after
    # discarding outliers — the final master bias frame.
    median_bias = np.ma.median(bias_clipped, axis=0)

    # Step 5: Fill masked (clipped) values with NaNs for compatibility with FITS format
    # FITS does not support masked arrays, so we replace invalid pixels with NaN.
    median_bias_filled = median_bias.filled(np.nan)

    # Step 6: Write the resulting 2D median bias to a FITS file
    # We use PrimaryHDU to save the array, and add a header comment for documentation.
    hdu = fits.PrimaryHDU(data=median_bias_filled)
    hdu.header['COMMENT'] = "Bias Frame, Sigma clipped (3-sigma)"
    hdu.writeto(median_bias_filename, overwrite=True)

    # Step 7: Return the processed bias frame (in case it's used later in-memory)
    return median_bias_filled

def create_median_dark(dark_list, bias_filename, median_dark_filename):
    """
    Create a median dark frame using bias correction, exposure-time normalization,
    sigma clipping, and save it to a FITS file.

    Parameters
    ----------
    dark_list : list of str
        A list of file paths to individual dark frames (FITS format).
    bias_filename : str
        Path to the previously created median bias frame (FITS file).
    median_dark_filename : str
        Name of the output FITS file where the median dark frame will be saved.

    Returns
    -------
    median_dark_filled : ndarray
        A 2D numpy array representing the final, normalized, sigma-clipped dark frame.
    """

    # Step 1: Read in each dark frame and extract the associated exposure time.
    # Exposure time is required to normalize the dark current per second.
    # Each dark frame is stored as a tuple: (image data, exposure time)
    dark_frames = []
    for dark_file in dark_list:
        with fits.open(dark_file) as hdul:
            data = hdul[0].data.astype('f4')  # Convert to float32
            exposure_time = hdul[0].header['EXPTIME']  # Extract exposure time from FITS header
            dark_frames.append((data, exposure_time))

    # Step 2: Load the median bias frame
    # This bias frame will be subtracted from each dark frame to remove readout noise.
    bias_frame = fits.getdata(bias_filename).astype('f4')

    # Step 3: Perform bias correction and normalize by exposure time
    # Subtract the bias frame and divide by exposure time to get dark current per second.
    corrected_frames = []
    for data, exposure_time in dark_frames:
        dark_corrected = (data - bias_frame) / exposure_time
        corrected_frames.append(dark_corrected)

    # Step 4: Stack all corrected dark frames into a 3D array
    # Sigma clip pixel-by-pixel to remove outliers such as cosmic rays.
    dark_stack = np.stack(corrected_frames, axis=0)
    dark_clipped = sigma_clip(dark_stack, cenfunc='median', sigma=3, axis=0)

    # Step 5: Compute the average dark current per pixel across clipped data
    # This produces the master dark frame, which can be scaled by exposure time later.
    median_dark = np.ma.mean(dark_clipped, axis=0)

    # Step 6: Fill masked (clipped) values with NaN to ensure FITS compatibility
    # Masked values from sigma clipping are filled for writing to disk.
    median_dark_filled = median_dark.filled(np.nan)

    # Step 7: Save the resulting dark frame to a FITS file
    # Add a FITS header comment for clarity and reproducibility.
    hdu = fits.PrimaryHDU(median_dark_filled)
    hdu.header['COMMENT'] = "Dark Frame, Sigma clipped (3-sigma)"
    hdu.writeto(median_dark_filename, overwrite=True)

    # Step 8: Return the final dark frame in case it's needed in memory
    return median_dark_filled

def create_median_flat(
    flat_list,
    bias_filename,
    median_flat_filename,
    dark_filename=None,
):
    """
    Create a normalized median flat frame using bias and (optionally) dark correction,
    sigma clipping, and save it to a FITS file.

    Parameters
    ----------
    flat_list : list of str
        A list of file paths to flat field images (FITS files).
    bias_filename : str
        Path to the median bias frame to subtract from the flats.
    median_flat_filename : str
        Path where the resulting normalized flat field will be saved.
    dark_filename : str, optional
        Path to the median dark frame (normalized per second) to subtract if provided.

    Returns
    -------
    normalized_flat : ndarray
        A 2D normalized flat frame ready for science frame correction.
    """

    # Step 1: Load flat field data and extract exposure times
    # Exposure times are needed for optional dark frame scaling.
    flat_frames = []
    exptimes = []
    for flat_file in flat_list:
        with fits.open(flat_file) as hdul:
            data = hdul[0].data.astype('f4')
            exptime = hdul[0].header['EXPTIME']
            flat_frames.append(data)
            exptimes.append(exptime)

    # Step 2: Load bias frame to remove fixed-pattern electronic noise
    bias_frame = fits.getdata(bias_filename).astype('f4')

    # Step 3: Subtract bias from each flat frame
    # This corrects for the electronic pedestal in CCD readouts.
    bias_corrected_frames = []
    for data in flat_frames:
        flat_corrected = data - bias_frame
        bias_corrected_frames.append(flat_corrected)

    # Step 3B: Optionally subtract scaled dark frame (if provided)
    # Each dark frame is scaled by the flat's exposure time to match units.
    if dark_filename is not None:
        dark_frame = fits.getdata(dark_filename).astype('f4')
        dark_corrected_frames = []
        for i, flat_corrected in enumerate(bias_corrected_frames):
            dark_scaled = dark_frame * exptimes[i]
            corrected = flat_corrected - dark_scaled
            dark_corrected_frames.append(corrected)
        corrected_frames = dark_corrected_frames
    else:
        corrected_frames = bias_corrected_frames

    # Step 4: Stack corrected flat frames into a 3D array
    # Sigma clipping removes pixel-level outliers like cosmic rays.
    flat_stack = np.stack(corrected_frames, axis=0)
    flat_clipped = sigma_clip(flat_stack, cenfunc="median", sigma=3, axis=0)

    # Step 5: Combine into median flat frame (pixel-wise average of good data)
    median_flat = np.ma.mean(flat_clipped, axis=0)

    # Step 6: Normalize the flat
    # This ensures the average pixel value is 1.0, so it preserves total flux when applied.
    normalized_flat = median_flat / np.ma.median(median_flat)

    # Step 7: Save normalized flat frame to FITS file
    # This file will be used to divide science frames and correct for pixel sensitivity.
    primary = fits.PrimaryHDU(data=normalized_flat.data)
    primary.header['COMMENT'] = 'Normalized median flat, bias corrected, sigma clipped'
    hdul = fits.HDUList([primary])
    hdul.writeto(median_flat_filename, overwrite=True)

    # Step 8: Return the normalized flat for potential in-memory use
    return normalized_flat

def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename="reduced_science.fits",
):
    """
    Perform full CCD reduction on a science frame using provided calibration files.

    Parameters
    ----------
    science_filename : str
        Path to the raw science frame to be reduced (FITS file).
    median_bias_filename : str
        Path to the master bias frame (created via create_median_bias).
    median_flat_filename : str
        Path to the normalized master flat field (created via create_median_flat).
    median_dark_filename : str
        Path to the dark frame normalized per second (created via create_median_dark).
    reduced_science_filename : str, optional
        Name for the output reduced science frame. Defaults to 'reduced_science.fits'.

    Returns
    -------
    reduced_science : ndarray
        A 2D numpy array representing the fully calibrated science frame.
    """

    # Step 1: Read in the raw science frame and extract its exposure time
    # We use float32 to standardize the datatype for consistent math operations
    with fits.open(science_filename) as hdul:
        science_data = hdul[0].data.astype('f4')
        exp_time = hdul[0].header['EXPTIME']  # Required for dark scaling
        science_header = hdul[0].header       # Preserve metadata in final output

    # Step 2: Load calibration frames from disk
    # These frames are assumed to be trimmed, corrected, and ready to use.
    median_bias = fits.getdata(median_bias_filename)
    median_flat = fits.getdata(median_flat_filename)
    median_dark = fits.getdata(median_dark_filename)

    # Step 3: Apply calibration steps in order
    # 1. Subtract the bias frame to remove readout noise offset
    # 2. Subtract the scaled dark frame (multiplied by science exposure time)
    # 3. Divide by the normalized flat frame to correct for pixel sensitivity
    dark_scaled = median_dark * exp_time
    reduced_science = (science_data - median_bias - dark_scaled) / median_flat

    # Step 4: Save the reduced science frame to a new FITS file
    # We reuse the original science header and document the reduction step.
    primary = fits.PrimaryHDU(data=reduced_science, header=science_header)
    primary.header['COMMENT'] = 'Reduced science frame: bias, dark, flat corrected'
    hdul = fits.HDUList([primary])
    hdul.writeto(reduced_science_filename, overwrite=True)

    # Step 5: Return the reduced frame in case it's needed for further processing
    return reduced_science

def run_reduction():
    # Define output filenames for calibration and diagnostics
    median_bias_fn = "median_bias.fits"
    median_dark_fn = "median_dark.fits"
    median_flat_fn = "median_flat.fits"

    # Find calibration and science files
    bias_list = sorted([f"../bias/{f}" for f in os.listdir("../bias") if f.endswith(".fits")])
    dark_list = sorted([f"../darks/{f}" for f in os.listdir("../darks") if f.endswith(".fits")])
    flat_list = sorted([f"../flats/{f}" for f in os.listdir("../flats") if f.endswith(".fits")])
    science_list = sorted([f"../GJ-486/{f}" for f in os.listdir("../GJ-486") if f.endswith(".fits")])
    science_list += sorted([f"../TOI-1199/{f}" for f in os.listdir("../TOI-1199") if f.endswith(".fits")])

    # Check for missing files and fail gracefully
    if not (bias_list and dark_list and flat_list and science_list):
        print("Missing one or more required input file types. Skipping reduction.")
        print(f"Bias files found: {len(bias_list)}")
        print(f"Dark files found: {len(dark_list)}")
        print(f"Flat files found: {len(flat_list)}")
        print(f"Science files found: {len(science_list)}")
        return
    
    # Print summary for debugging
    print(f"Found {len(bias_list)} bias files")
    print(f"Found {len(dark_list)} dark files")
    print(f"Found {len(flat_list)} flat files")
    print(f"Found {len(science_list)} science files")

    # Create calibration frames
    median_bias = create_median_bias(bias_list, median_bias_fn)
    median_dark = create_median_dark(dark_list, median_bias_fn, median_dark_fn)
    median_flat = create_median_flat(flat_list, median_bias_fn, median_flat_fn, dark_filename=median_dark_fn)

    # Reduce Science Frames
    reduced_fns = []
    for i, sci_fn in enumerate(science_list):
        base_name = os.path.basename(sci_fn).replace(".fits", "")
        if "GJ-486" in sci_fn:
            subdir = os.path.join("reductions", "GJ-486")
        elif "TOI-1199" in sci_fn:
            subdir = os.path.join("reductions", "TOI-1199")
        else:
            subdir = "reductions"
        os.makedirs(subdir, exist_ok=True)
        out_fn = os.path.join(subdir, f"{base_name}_reduced.fits")
        print(f"Reducing science frame: {sci_fn}")
        reduce_science_frame(
            sci_fn,
            median_bias_fn,
            median_flat_fn,
            median_dark_fn,
            reduced_science_filename=out_fn
        )
        reduced_fns.append(out_fn)


# Run the reduction pipeline if this script is executed directly
if __name__ == "__main__":
    run_reduction()
    print("Reduction pipeline completed.")