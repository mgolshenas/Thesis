import os, re, shutil
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Module_build_regression_matrix import build_regression_matrix  

start = time.perf_counter()

# ------
raster_dir = r"C:\Users\M\Downloads\Output"
filtered_dir = r"C:\Users\M\Downloads\Output_filtered"
raster_path = r"C:\Users\M\Downloads\DEM01.gtif.tif"

# --- STEP 1: Prepare filtered_dir ---
os.makedirs(filtered_dir, exist_ok=True)

# --- STEP 2: Generate sampling points ---
with rasterio.open(raster_path) as src:
    bounds, crs = src.bounds, src.crs
num_points = 30
np.random.seed(42)
random_x = np.random.uniform(bounds.left, bounds.right, num_points)
random_y = np.random.uniform(bounds.bottom, bounds.top, num_points)
subset_pixels_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(random_x, random_y)], crs=crs)

# --- STEP 3: Loop over digit positions and values ---
for digit_pos in range(1, 5):  # Positions 1 to 4
    for target_value in ["0", "1"]:

        # Clear filtered directory
        [os.remove(os.path.join(filtered_dir, f)) for f in os.listdir(filtered_dir) if os.path.isfile(os.path.join(filtered_dir, f))]

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
                    shutil.copy2(os.path.join(raster_dir, f), os.path.join(filtered_dir, f))
                    filtered_files.append(os.path.join(filtered_dir, f))

        print(f"[Digit {digit_pos}, Value '{target_value}'] → Filtered {len(filtered_files)} files")

        if not filtered_files:
            print("No matching files. Skipping this round.")
            continue

        # --- STEP 4: Build regression matrix ---
        regression_matrix = build_regression_matrix(filtered_dir, subset_pixels_gdf)
        print(regression_matrix.head())
        print("Regression matrix shape:", regression_matrix.shape)

        # --- STEP 5: Sample target values ---
        with rasterio.open(raster_path) as src:
            coords = [(point.x, point.y) for point in subset_pixels_gdf.geometry]
            y = np.array(list(src.sample(coords))).flatten()

        # --- STEP 6: Prepare feature matrix ---
        X = regression_matrix.select_dtypes(include=[np.number])
        mask = ~np.isnan(y)
        X = X.iloc[mask]
        y_clean = y[mask]

        # Skip if empty after filtering
        if X.empty or y_clean.size == 0:
            print("Empty matrix after filtering NaNs. Skipping.")
            continue

        # --- STEP 7: Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)

        # --- STEP 8: Train Random Forest ---
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [150, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # --- STEP 9: Evaluate model ---
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n[Digit {digit_pos}, Value '{target_value}'] Evaluation:")
        print("Best Parameters:", grid_search.best_params_)
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}\n")

end = time.perf_counter()
print(f"Total execution time: {end - start:.2f} sec")
