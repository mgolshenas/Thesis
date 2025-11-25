

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import rasterio

# --- Function to sample raster values at given points ---
def sample_raster_at_points(raster_path, gdf_points):
    with rasterio.open(raster_path) as src:
        coords = [(point.x, point.y) for point in gdf_points.geometry]
        values = list(src.sample(coords))
        return np.array(values).flatten()

# --- Step 1: Sample target values from DEM raster ---
y = sample_raster_at_points(raster_path, subset_pixels_gdf)

# --- Step 2: Clean feature matrix by keeping numeric columns only ---
X = regression_matrix.select_dtypes(include=[np.number])

# --- Step 3: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Define Random Forest and hyperparameter grid ---
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# --- Step 5: Fit the model ---
grid_search.fit(X_train, y_train)

# --- Step 6: Predict and evaluate ---
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
