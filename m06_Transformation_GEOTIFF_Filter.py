import numpy as np
import rasterio
import os
import gc
import configparser

from Module_process_and_display_fourier_zone_filter import process_and_display_fourier_zone_filter

# --- Load Config ---
config = configparser.ConfigParser()
config.read('config.ini')
print("Config sections found:", config.sections())
seed = 42
if 'Random' in config and 'seed' in config['Random']:
    np.random.seed(int(config['Random']['seed']))


# Validate config
try:
    input_path = config['Paths']['input_file']
    output_folder = config['Paths']['output_folder']
    zone_codes = config['Fourier']['zone_codes'].split(',')
except KeyError as e:
    raise KeyError(f"Missing config entry: {e}")

# Make output folder
os.makedirs(output_folder, exist_ok=True)

# Format zone codes as 4-bit binary strings (excluding '0000')
zone_codes = [z.zfill(4) for z in zone_codes if z.strip() != '0000']

for zone_code in zone_codes:
    print(f"zone_code: {zone_code}, length: {len(zone_code)}")

# --- Transformation Functions ---
def transform_multiplication(img, b, c):
    return b * img + c

def transform_quadratic(img, c):
    return np.square(img) + c

def transform_logarithmic(img, c):
    return np.log1p(np.abs(img)) + c  # log(1 + |x|) + c

def transform_sine(img, c):
    return np.sin(img) + c

# --- Master Function to Apply All Transformations with Random b, c ---
def apply_all_transformations(img):
    b = np.random.uniform(0.5, 3.0)
    c = np.random.uniform(0.5, 3.0)

    return {
        "mult": transform_multiplication(img, b, c),
        "quad": transform_quadratic(img, c),
        "log": transform_logarithmic(img, c),
        "sin": transform_sine(img, c)
    }, b, c  # Return b and c for logging if needed

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

            # Apply and save each transformation with random b, c
            transformed, b, c = apply_all_transformations(arr)

            for tname, timg in transformed.items():
                tpath = os.path.join(output_folder, f"{name}_{tname}.tif")
                tprofile = base_profile.copy()
                tprofile.update(dtype=str(timg.dtype), count=1)

                with rasterio.open(tpath, 'w', **tprofile) as dst:
                    dst.write(timg, 1)
                print(f"Saved: {tpath} | b={b:.3f}, c={c:.3f}")

        gc.collect()
