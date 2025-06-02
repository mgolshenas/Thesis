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

zone_codes = [format(i, '04b') for i in range(1, 16)]  # '0001' to '1111' excluding '0000'

# --- Transformation Functions ---
def transform_multiplication(img, b=2.0, c=1.0):
    return b * img + c

def transform_quadratic(img, c=1.0):
    return np.square(img) + c

def transform_logarithmic(img, c=1.0):
    return np.log1p(np.abs(img)) + c  # log(1 + |x|) + c

def transform_sine(img, c=1.0):
    return np.sin(img) + c

# --- Master Function to Apply All Transformations ---
def apply_all_transformations(img, c=1.0, b=2.0):
    return {
        "mult": transform_multiplication(img, b, c),
        "quad": transform_quadratic(img, c),
        "log": transform_logarithmic(img, c),
        "sin": transform_sine(img, c)
    }

# --- Read and Process Image ---
with rasterio.open(input_path) as src:
    image = src.read(1)
    base_profile = src.profile.copy()

    for zone_code in zone_codes:
        # Apply Fourier zone filter
        filtered_img, filtered_spectrum = process_and_display_fourier_zone_filter(image, zone_code)

        # Base output names
        base_img_name = f"filtered_img_{zone_code}"
        base_spec_name = f"spectrum_{zone_code}"

        for name, arr in [(base_img_name, filtered_img), (base_spec_name, filtered_spectrum)]:
            # Write original filtered output
            path = os.path.join(output_folder, f"{name}.tif")
            profile = base_profile.copy()
            profile.update(dtype=str(arr.dtype), count=1)

            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(arr, 1)
            print(f"Saved: {path}")

            # Apply and save each transformation
            transformed = apply_all_transformations(arr)

            for tname, timg in transformed.items():
                tpath = os.path.join(output_folder, f"{name}_{tname}.tif")
                tprofile = base_profile.copy()
                tprofile.update(dtype=str(timg.dtype), count=1)

                with rasterio.open(tpath, 'w', **tprofile) as dst:
                    dst.write(timg, 1)
                print(f"Saved: {tpath}")

        gc.collect()
