# In[1]:


import os
import rasterio
import geopandas as gpd
import pandas as pd

def sample_rasters_at_points(outfolder: str, gdf_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Sample multiple raster files at specified point locations.

    Parameters:
        outfolder (str): Path to folder containing .tif raster files.
        gdf_points (GeoDataFrame): GeoDataFrame with point geometries.

    Returns:
        GeoDataFrame: Copy of gdf_points with raster values appended as new columns.
    """
    raster_files = [f for f in os.listdir(outfolder) if f.endswith('.tif')]
    coords = [(geom.x, geom.y) for geom in gdf_points.geometry]

    # Collect all new columns in a dict first
    raster_data = {}

    for raster_file in raster_files:
        raster_path = os.path.join(outfolder, raster_file)
        with rasterio.open(raster_path) as src:
            values = [val[0] if val is not None else None for val in src.sample(coords)]
            col_name = os.path.splitext(raster_file)[0]
            raster_data[col_name] = values

    # Convert dict to DataFrame and concatenate with original GeoDataFrame
    df_raster = pd.DataFrame(raster_data)
    gdf_result = pd.concat([gdf_points.reset_index(drop=True), df_raster], axis=1)

    return gdf_result





# In[]:


