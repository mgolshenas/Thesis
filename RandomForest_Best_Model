from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Split the data BEFORE grid search to avoid data leakage ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define the base model ---
rf = RandomForestRegressor(random_state=42)

# --- Define grid of hyperparameters to search ---
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt']
}

# --- Setup GridSearchCV ---
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# --- Fit on training data only ---
grid_search.fit(X_train, y_train)

# --- Use the best estimator ---
best_rf = grid_search.best_estimator_

print("Best Hyperparameters:")
print(grid_search.best_params_)

# --- Predict on the test set ---
y_pred = best_rf.predict(X_test)

# --- Evaluate the model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)




print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
