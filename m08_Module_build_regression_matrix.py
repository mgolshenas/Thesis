
# In[1]:

import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def build_regression_matrix(raster_folder, pixel_gdf):
    """
    Constructs a regression matrix from rasters for given pixel locations.

    Parameters:
    -----------
    raster_folder : str
        Path to the folder containing raster images (.tif).
    pixel_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing Point geometries.

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with original pixel points and one column per raster.
    """
    if 'geometry' not in pixel_gdf:
        raise ValueError("pixel_gdf must contain a 'geometry' column with Point geometries.")
    
    # Ensure it's a GeoDataFrame with proper CRS
    pixel_gdf = pixel_gdf.copy()
    raster_files = [f for f in os.listdir(raster_folder) if f.lower().endswith('.tif')]
    
    if not raster_files:
        raise FileNotFoundError("No .tif raster files found in the specified folder.")

    # Output DataFrame starts with geometry
    result_df = pixel_gdf.copy()

    for raster_file in raster_files:
        raster_path = os.path.join(raster_folder, raster_file)
        with rasterio.open(raster_path) as src:
            # Reproject points if CRS differs
            if pixel_gdf.crs != src.crs:
                points = pixel_gdf.to_crs(src.crs)
            else:
                points = pixel_gdf

            # Extract raster values at point locations
            coords = [(pt.x, pt.y) for pt in points.geometry]
            values = [val[0] if val is not None else None for val in src.sample(coords)]
            band_name = os.path.splitext(raster_file)[0]  # filename without extension
            result_df[band_name] = values

    return result_df



# In[]:
