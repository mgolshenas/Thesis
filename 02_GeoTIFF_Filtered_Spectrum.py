
import numpy as np
import rasterio
import os
import gc
from Module_process_and_display_fourier_zone_filter import process_and_display_fourier_zone_filter

# Input/output paths
input_path = r"C:\Users\M\Downloads\DEM01.gtif.tif"
input_path = input_path.encode('ascii', 'ignore').decode()
output_folder = r"C:\Users\M\Downloads\Output"
os.makedirs(output_folder, exist_ok=True)

zone_codes = [format(i, '04b') for i in range(1, 16)]

# Read source image
with rasterio.open(input_path) as src:
    image = src.read(1)
    base_profile = src.profile.copy()

    for zone_code in zone_codes:
        # Get filtered image and spectrum
        filtered_img, filtered_spectrum = process_and_display_fourier_zone_filter(image, zone_code)

        # Save filtered image
        img_profile = base_profile.copy()
        img_profile.update(dtype=str(filtered_img.dtype), count=1)
        output_img_path = os.path.join(output_folder, f"filtered_img_{zone_code}.tif")
        with rasterio.open(output_img_path, 'w', **img_profile) as dst:
            dst.write(filtered_img, 1)

        # Save filtered spectrum
        spec_profile = base_profile.copy()
        spec_profile.update(dtype=str(filtered_spectrum.dtype), count=1)
        output_spec_path = os.path.join(output_folder, f"spectrum_{zone_code}.tif")
        with rasterio.open(output_spec_path, 'w', **spec_profile) as dst:
            dst.write(filtered_spectrum, 1)

        print(f"Saved: {output_img_path} and {output_spec_path}")
        gc.collect()
