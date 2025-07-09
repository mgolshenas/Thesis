 # Fourier-Based Image Processing Modules

## 1. Module_Fourier_Spectrum

**Purpose**  
Reads an image (`Earth.png`), converts it to grayscale, and computes its 2D Fourier Transform. It displays both the image and its frequency spectrum.

**Main Functionalities**
- `cv2.imread()` to read the image.
- `np.fft.fft2()` and `np.fft.fftshift()` to compute the FFT.

**Displays**
- Grayscale image.
- Log-magnitude Fourier spectrum.

---

## 2. GeoTIFF_Filtered_Spectrum *(Requires `rasterio` library)*

**Purpose**  
Applies zone-based frequency filtering to a GeoTIFF image (`DEM01.gtif.tif`), saving filtered outputs for multiple frequency masks.

**Process Overview**
- Loads image using `rasterio`.
- Creates an output folder.
- Generates 4-bit binary zone codes (from `'0001'` to `'1111'`) to enable selective frequency zones.
- For each `zone_code`, it:
  - Calls `process_and_display_fourier_zone_filter()` from the corresponding module.
  - Saves:
    - `filtered_img_{zone_code}.tif` (spatial image)
    - `spectrum_{zone_code}.tif` (Fourier spectrum)

---

## 3. Module_Fourier_zone_filter

**Purpose**  
Provides a reusable function to apply a concentric-zone frequency filter using a binary zone code.

```python
def fourier_zone_filter(image: np.ndarray, zone_code: str) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters**
- `image`: 2D grayscale NumPy array
- `zone_code`: 4-character binary string (e.g., `'1010'`)

**Returns**
- `filtered_img`: Filtered image in the spatial domain.
- `filtered_spectrum`: Log-magnitude spectrum after filtering.

---

## 4. Module_process_and_display_fourier_zone_filter

**Purpose**  
Handles the full processing pipeline: FFT, zone filtering, inverse FFT, and spectrum visualization.

```python
def process_and_display_fourier_zone_filter(image: np.ndarray, zone_code: str) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters**
- `image`: Grayscale image (2D NumPy array)
- `zone_code`: 4-digit binary string controlling which radial frequency bands to preserve

**Steps**
- Normalize and convert image
- Compute FFT and shift spectrum
- Generate frequency mask from `zone_code`
- Apply mask and inverse FFT

**Returns**
- Filtered image and spectrum for display

---

## 5. The_inverse_Fourier_Transform

**Purpose**  
Visualizes how different frequency components reconstruct an image over time.

**Main Functionalities**
- Starts with low-frequency pairs and adds higher frequencies step-by-step.
- Shows how each pair contributes to the final image.
- Educational visualization for understanding image reconstruction from FFT.

---

## 6. Transformation_GEOTIFF_Filter *(Requires `rasterio` library)*

**Purpose**  
Processes a GeoTIFF using Fourier zone filtering. For each of 15 zone codes (`'0001'` to `'1111'`, excluding `'0000'`), it:

- Applies `process_and_display_fourier_zone_filter`
- For each result, performs 4 transformations:

  - Multiplication: `b * x + c`
  - Quadratic: `x² + c`
  - Logarithmic: `log(1 + |x|) + c`
  - Sine: `sin(x) + c`

**Output**
- 64 total feature images (4 transforms × 2 output types × 8 zone codes)

---

## 7. Raster Sampling Function

**Purpose**  
Sample values from multiple raster (`.tif`) files at given point locations.

```python
def sample_rasters_at_points(outfolder: str, gdf_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
```

**Inputs**
- `outfolder`: Folder containing raster files
- `gdf_points`: GeoDataFrame with point geometries

**Process**
1. List all `.tif` raster files in the folder.
2. Extract `(x, y)` coordinates from the points.
3. For each raster, sample values at these points.
4. Store sampled values in a dictionary keyed by raster filename.
5. Convert the dictionary to a DataFrame.
6. Concatenate the raster values DataFrame with the original GeoDataFrame.

**Output**
- GeoDataFrame with original points and one column per raster containing sampled values


## 8. Regression Matrix Builder Using First N Points
**Purpose**
Extracts the coordinates of the first n valid (non-NoData) pixels from a raster and builds a regression matrix by sampling values from other raster layers at those points.

**Inputs**

raster_path: Path to the reference raster file (target DEM).

raster_folder: Folder containing input rasters for features.

num_points: Number of pixel points to extract (first N).

**Process**

Opens the reference raster using rasterio.

Filters out NoData pixels.

Converts valid pixel indices to (x, y) coordinates.

Constructs a GeoDataFrame from the first n points.

Passes these points to build_regression_matrix() to create the feature matrix.

**Output**

- regression_matrix: DataFrame of features at the selected pixel locations.

- subset_pixels_gdf: GeoDataFrame of the sampled pixel coordinates.





      
