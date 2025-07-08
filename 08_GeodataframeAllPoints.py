import os
import numpy as np
import rasterio
import pandas as pd

# --- Load reference raster to define CRS, transform, and shape ---
raster_path = r"C:\Users\MH\Downloads\MachineLearning\DEM01.gtif.tif"

with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    transform = src.transform
    width = src.width
    height = src.height
    profile = src.profile

# --- Read all raster files in the output directory ---
raster_dir = r"C:\Users\MH\Downloads\MachineLearning\Output"
raster_files = [f for f in os.listdir(raster_dir) if f.endswith('.tif')]

# Dictionary to hold raster arrays by filename (or any name you want)
raster_arrays = {}

for filename in raster_files:
    path = os.path.join(raster_dir, filename)
    with rasterio.open(path) as src:
        # Make sure all rasters align (same shape and transform)
        assert src.width == width and src.height == height, "Raster dimension mismatch"
        assert src.transform == transform, "Raster transform mismatch"
        
        # Read first band only (modify if you want multiple bands)
        raster_arrays[filename] = src.read(1)

# --- Combine all rasters into a DataFrame ---
# Flatten each raster to 1D array of pixel values
data = {name: arr.flatten() for name, arr in raster_arrays.items()}

df = pd.DataFrame(data)

# Optionally, add pixel coordinates (x, y) for each pixel center
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
x_coords, y_coords = rasterio.transform.xy(transform, rows, cols, offset='center')
df['x'] = np.array(x_coords).flatten()
df['y'] = np.array(y_coords).flatten()

# Now df has columns: one per raster file plus x,y coords per pixel

# --- Preview ---
print(df.head())
print(df.shape)

