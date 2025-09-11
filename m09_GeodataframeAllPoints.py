import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio

from Module_build_regression_matrix import build_regression_matrix  

def get_num_points_pixel_points(raster_path, num_points=None):
    """
    Extracts the coordinates of valid pixels (non-nodata) in a raster.
    Optionally limits the output to the first `num_points`.

    Parameters:
    -----------
    raster_path : str
        Path to the raster file.
    num_points : int, optional
        Number of points to return from the beginning.

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame of pixel center points with CRS matching the raster.
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs

        # Mask valid values (not equal to NoData)
        if src.nodata is not None:
            mask = band != src.nodata
        else:
            mask = ~np.isnan(band)

        rows, cols = np.where(mask)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        points = [Point(x, y) for x, y in zip(xs, ys)]

        gdf = gpd.GeoDataFrame(geometry=points, crs=crs)

    if num_points is not None:
        gdf = gdf.head(num_points)

    return gdf

# --- INPUTS ---
raster_path = r"C:\Users\MH\Downloads\MachineLearning\DEM01.gtif.tif"
raster_folder = r"C:\Users\MH\Downloads\MachineLearning\Output"
num_points = 900  # Use only the first num_points valid pixel points

# --- STEP 1: Create GeoDataFrame of the first N raster pixels ---
subset_pixels_gdf = get_num_points_pixel_points(raster_path, num_points=num_points)

# --- STEP 2: Use the module to build regression matrix ---
regression_matrix = build_regression_matrix(raster_folder, subset_pixels_gdf)

# --- STEP 3: Output ---
print(regression_matrix.head())
print(subset_pixels_gdf.head())

# --- Preview ---
print(regression_matrix.shape)
