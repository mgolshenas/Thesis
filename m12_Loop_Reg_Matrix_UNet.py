import os, re, shutil
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Module_build_regression_matrix import build_regression_matrix  

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, concatenate, Lambda

start = time.perf_counter()

# ---- Paths ----
raster_dir = r"C:\\Users\\M\\Downloads\\Output"
filtered_dir = r"C:\\Users\\M\\Downloads\\Output_filtered"
raster_path = r"C:\\Users\\M\\Downloads\\DEM01.gtif.tif"

# --- STEP 1: Prepare filtered_dir ---
os.makedirs(filtered_dir, exist_ok=True)

# --- STEP 2: Generate sampling points ---
with rasterio.open(raster_path) as src:
    bounds, crs = src.bounds, src.crs

num_points = 10000
np.random.seed(42)
random_x = np.random.uniform(bounds.left, bounds.right, num_points)
random_y = np.random.uniform(bounds.bottom, bounds.top, num_points)
subset_pixels_gdf = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(random_x, random_y)], 
    crs=crs
)

# --- STEP 3: Load full DEM ---
with rasterio.open(raster_path) as src:
    elevation = src.read(1)

# --- STEP 4: Loop over digit positions and values ---
patch_size = 16
half_patch = patch_size // 2

for digit_pos in range(1, 5):
    for target_value in ["0", "1"]:

        # Clear filtered directory
        [os.remove(os.path.join(filtered_dir, f)) 
         for f in os.listdir(filtered_dir) 
         if os.path.isfile(os.path.join(filtered_dir, f))]

        # Filter rasters
        keep_digits = {digit_pos: target_value}
        pattern = re.compile(r"filtered_img_(\d{4})_.*")
        filtered_files = []
        for f in os.listdir(raster_dir):
            m = pattern.match(f)
            if m:
                code = m.group(1)
                conditions = [code[i - 1] == d for i, d in keep_digits.items()]
                if all(conditions):
                    shutil.copy2(os.path.join(raster_dir, f), 
                                 os.path.join(filtered_dir, f))
                    filtered_files.append(os.path.join(filtered_dir, f))

        print(f"[Digit {digit_pos}, Value '{target_value}'] → Filtered {len(filtered_files)} files")
        if not filtered_files:
            print("No matching files. Skipping this round.")
            continue

        # --- STEP 5: Build regression matrix (optional) ---
        regression_matrix = build_regression_matrix(filtered_dir, subset_pixels_gdf)
        print(regression_matrix.head())
        print("Regression matrix shape:", regression_matrix.shape)

        # --- STEP 6: Extract 2D patches as input and output ---
        X_patches = []
        y_patches = []
        for point in subset_pixels_gdf.geometry:
            row, col = src.index(point.x, point.y)
            patch = elevation[
                row-half_patch:row+half_patch,
                col-half_patch:col+half_patch
            ]
            if patch.shape == (patch_size, patch_size):
                X_patches.append(patch)
                y_patches.append(patch)  # output is the full patch

        X_patches = np.array(X_patches)[..., np.newaxis]
        y_patches = np.array(y_patches)[..., np.newaxis]

        if X_patches.size == 0:
            print("No valid patches. Skipping this round.")
            continue

        # --- STEP 7: Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_patches, y_patches, test_size=0.2, random_state=42
        )

        # --- STEP 8: Define Encoder-Decoder (U-Net style) ---
        inputs = Input(shape=(patch_size, patch_size, 1))

        # Encoder
        conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2,2))(conv1)
        drop1 = Dropout(0.5)(conv1)

        conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2,2))(conv2)
        drop2 = Dropout(0.5)(conv2)

        # Decoder
        up1 = UpSampling2D((2,2))(drop2)
        up1 = Conv2D(64, (3,3), activation='relu', padding='same')(up1)

        # Resize to match encoder feature map
        up1_resized = Lambda(lambda x: tf.image.resize(x, (conv2.shape[1], conv2.shape[2])))(up1)
        merge1 = concatenate([conv2, up1_resized])
        conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(merge1)
        conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(conv3)

        up2 = UpSampling2D((2,2))(conv3)
        up2 = Conv2D(32, (3,3), activation='relu', padding='same')(up2)
        up2_resized = Lambda(lambda x: tf.image.resize(x, (conv1.shape[1], conv1.shape[2])))(up2)
        merge2 = concatenate([conv1, up2_resized])
        conv4 = Conv2D(32, (3,3), activation='relu', padding='same')(merge2)
        conv4 = Conv2D(32, (3,3), activation='relu', padding='same')(conv4)

        outputs = Conv2D(1, (1,1), activation='linear', padding='same')(conv4)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # --- STEP 9: Train ---
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=16,
            verbose=1
        )

        # --- STEP 10: Evaluate ---
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.flatten(), y_pred.flatten())

        print(f"\n[Digit {digit_pos}, Value '{target_value}'] Evaluation (Encoder-Decoder 2D CNN):")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}\n")

end = time.perf_counter()
print(f"Total execution time: {end - start:.2f} sec")
