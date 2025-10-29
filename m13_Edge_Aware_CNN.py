import os
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping

from Module_build_regression_matrix import build_regression_matrix

# --- TIMER START ---
start = time.perf_counter()

# ---- Paths ----
raster_dir = r"C:\\Users\\M\\Downloads\\Output"
raster_path = r"C:\\Users\\M\\Downloads\\DEM01.gtif.tif"
base_output_path = r"C:\\Users\\M\\Downloads\\DEM01_predicted"

# --- Load list of rasters ---
raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith(".tif")]
if len(raster_files) == 0:
    raise FileNotFoundError("No raster files found in Output folder.")

# --- Load DEM ---
with rasterio.open(raster_path) as src:
    elevation = src.read(1).astype(np.float32)
    profile = src.profile
    bounds, crs = src.bounds, src.crs

elevation = np.where(np.isfinite(elevation), elevation, np.nan)
y_min, y_max = np.nanmin(elevation), np.nanmax(elevation)

# --- Generate random sample points ---
num_points = 1000
np.random.seed(42)
random_x = np.random.uniform(bounds.left, bounds.right, num_points)
random_y = np.random.uniform(bounds.bottom, bounds.top, num_points)
subset_pixels_gdf = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(random_x, random_y)], crs=crs
)

# --- Patch size ---
patch_size = 16
half_patch = patch_size // 2

# --- STEP 3: Build regression matrix ---
regression_matrix = build_regression_matrix(raster_dir, subset_pixels_gdf)
print("Regression matrix shape:", regression_matrix.shape)

# --- STEP 4: Build stack from all raster files ---
stack_list = []
for f in sorted(raster_files):
    with rasterio.open(f) as src:
        stack_list.append(src.read(1))
filtered_stack = np.stack(stack_list, axis=-1)
print("Stack shape:", filtered_stack.shape)

# --- Normalize ---
X_min, X_max = np.nanmin(filtered_stack), np.nanmax(filtered_stack)
X_norm = (filtered_stack - X_min) / (X_max - X_min + 1e-8)
y_norm = (elevation - y_min) / (y_max - y_min + 1e-8)

# --- Edge-aware patch extraction helper ---
def get_edge_aware_patch(X, r, c, patch_size):
    half = patch_size // 2
    r_start, r_end = r - half, r + half
    c_start, c_end = c - half, c + half
    patch = X[max(r_start, 0):min(r_end, X.shape[0]),
              max(c_start, 0):min(c_end, X.shape[1]), :]
    pad_top = max(0, -r_start)
    pad_bottom = max(0, r_end - X.shape[0])
    pad_left = max(0, -c_start)
    pad_right = max(0, c_end - X.shape[1])
    patch = np.pad(patch,
                   ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                   mode='reflect')
    return patch

# --- STEP 6: Extract patches (edge-aware) ---
X_patches, y_patches = [], []
with rasterio.open(raster_path) as src:
    for point in subset_pixels_gdf.geometry:
        row, col = src.index(point.x, point.y)
        patch_input = get_edge_aware_patch(X_norm, row, col, patch_size)
        patch_output = get_edge_aware_patch(y_norm[..., np.newaxis], row, col, patch_size)
        X_patches.append(patch_input)
        y_patches.append(patch_output)

X_patches = np.array(X_patches, dtype=np.float32)
y_patches = np.array(y_patches, dtype=np.float32)
print(f"Total patches extracted: {X_patches.shape[0]}")

# --- Train/test split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_patches, y_patches, test_size=0.2, random_state=42
)

# --- CNN Model ---
input_channels = X_patches.shape[-1]
inputs = Input(shape=(patch_size, patch_size, input_channels))

conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)

up1 = UpSampling2D((2, 2))(pool2)
merge1 = concatenate([conv2, up1])
conv3 = Conv2D(32, (3, 3), activation="relu", padding="same")(merge1)

up2 = UpSampling2D((2, 2))(conv3)
merge2 = concatenate([conv1, up2])
conv4 = Conv2D(16, (3, 3), activation="relu", padding="same")(merge2)

outputs = Conv2D(1, (1, 1), activation="linear", padding="same")(conv4)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("Training CNN model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=50, batch_size=16, verbose=1, callbacks=[early_stop])

# --- Predict Entire Raster (edge-aware, batched) ---
rows, cols = elevation.shape
predicted_raster = np.zeros((rows, cols), dtype=np.float32)
batch_size_pred = 512
batch_patches, pixel_positions = [], []

print("Predicting entire raster (edge-aware)...")
for r in range(rows):
    if r % 100 == 0:
        print(f"Processing row {r}/{rows}")
    for c in range(cols):
        patch = get_edge_aware_patch(X_norm, r, c, patch_size)
        batch_patches.append(patch)
        pixel_positions.append((r, c))
        if len(batch_patches) == batch_size_pred:
            preds = model.predict(np.array(batch_patches), verbose=0)
            for (rr, cc), p in zip(pixel_positions, preds):
                predicted_raster[rr, cc] = p[patch_size//2, patch_size//2, 0]
            batch_patches.clear()
            pixel_positions.clear()

# Remaining pixels
if batch_patches:
    preds = model.predict(np.array(batch_patches), verbose=0)
    for (rr, cc), p in zip(pixel_positions, preds):
        predicted_raster[rr, cc] = p[patch_size//2, patch_size//2, 0]

# --- Denormalize + Evaluate ---
predicted_raster = predicted_raster * (y_max - y_min) + y_min
valid_mask = np.isfinite(elevation)
mse = mean_squared_error(elevation[valid_mask], predicted_raster[valid_mask])
rmse = np.sqrt(mse)
r2 = r2_score(elevation[valid_mask], predicted_raster[valid_mask])

print("\nEvaluation Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# --- Save Output ---
output_path = f"{base_output_path}.tif"
profile.update(dtype=rasterio.float32, count=1, compress="lzw")
with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(predicted_raster, 1)

print(f"Saved predicted DEM to: {output_path}")

# --- TIMER END ---
end = time.perf_counter()
print(f"\nTotal execution time: {end - start:.2f} sec")
