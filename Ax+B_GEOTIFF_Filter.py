import numpy as np
import rasterio
import os
import gc
from Module_process_and_display_fourier_zone_filter import process_and_display_fourier_zone_filter

# Constants for transformation: b·x + c
b = 2.0  # Multiplication factor
c = 0.5  # Offset

# Input/output paths
input_path = r"C:\Users\M\Downloads\DEM01.gtif.tif"
input_path = input_path.encode('ascii', 'ignore').decode()
output_folder = r"C:\Users\M\Downloads\Output"
os.makedirs(output_folder, exist_ok=True)

# Generate 4-bit zone codes from 0001 to 1111
zone_codes = [format(i, '04b') for i in range(1, 16)]

# Read source image
with rasterio.open(input_path) as src:
    image = src.read(1).astype(np.float32)
    profile = src.profile.copy()

    # Apply transformation BEFORE FFT
    image = b * image + c

    for zone_code in zone_codes:
        # Unpack filtered image and spectrum from the function
        filtered_img, filtered_spectrum = process_and_display_fourier_zone_filter(image, zone_code)

        # Update profile to ensure correct output type
        profile.update(dtype=rasterio.float32, count=1)

        # Save filtered spatial-domain image
        output_img_path = os.path.join(output_folder, f"filtered_img_{zone_code}.tif")
        with rasterio.open(output_img_path, 'w', **profile) as dst:
            dst.write(filtered_img, 1)

        # Save filtered frequency-domain spectrum image
        output_spec_path = os.path.join(output_folder, f"spectrum_{zone_code}.tif")
        with rasterio.open(output_spec_path, 'w', **profile) as dst:
            dst.write(filtered_spectrum, 1)

        print(f"Saved: {output_img_path} and {output_spec_path}")
        gc.collect()
