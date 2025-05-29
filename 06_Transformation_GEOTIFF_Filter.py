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

# Simple transformations
def apply_transformations(img, c=1.0, b=2.0):
    img = img.astype(np.float32)
    return {
        "mult": b * img + c,
        "quad": np.square(img) + c,
        "log": np.log1p(np.abs(img)) + c,  # log1p for stability plus c as costant number avoid negative/zero
        "sin": np.sin(img) + c
    }

# Read source image
with rasterio.open(input_path) as src:
    image = src.read(1)
    profile = src.profile.copy()

    for zone_code in zone_codes:
        # Apply Fourier zone filter
        filtered_img, filtered_spectrum = process_and_display_fourier_zone_filter(image, zone_code)

        # Update profile for float32 output
        profile.update(dtype=rasterio.float32, count=1)

        # Save original filtered images
        base_img_name = f"filtered_img_{zone_code}"
        base_spec_name = f"spectrum_{zone_code}"

        for name, arr in [(base_img_name, filtered_img), (base_spec_name, filtered_spectrum)]:
            path = os.path.join(output_folder, f"{name}.tif")
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            print(f"Saved: {path}")

            # Apply and save each transformation
            transformed = apply_transformations(arr)

            for tname, timg in transformed.items():
                tpath = os.path.join(output_folder, f"{name}_{tname}.tif")
                with rasterio.open(tpath, 'w', **profile) as dst:
                    dst.write(timg.astype(np.float32), 1)
                print(f"Saved: {tpath}")

        gc.collect()
