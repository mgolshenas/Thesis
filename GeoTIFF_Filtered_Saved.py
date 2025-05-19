import numpy as np
import rasterio
import os
import gc

# Import the 'fourier_concentric_zone_filter' function from 'fourier_zone_filter.py'
from Module_Fourier_zone_filter import fourier_concentric_zone_filter

# Paths
input_path = r"C:\Users\M\Downloads\DEM01.gtif.tif"
input_path = input_path.encode('ascii', 'ignore').decode()

output_folder = r"C:\Users\M\Downloads\Output"
os.makedirs(output_folder, exist_ok=True)

# Generate 4-digit binary zone codes (excluding 0000)
zone_codes = [format(i, '04b') for i in range(1, 16)]

# Process image
with rasterio.open(input_path) as src:
    image = src.read(1)

    for zone_code in zone_codes:
        filtered_img = fourier_concentric_zone_filter(image, zone_code)

        output_path = os.path.join(output_folder, f"filtered_img_{zone_code}.tif")

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(filtered_img, 1)

        print(f"Saved: {output_path}")
        gc.collect()

    
