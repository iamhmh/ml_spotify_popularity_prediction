import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# === LINEAR REGRESSION === #
print("\n--- Linear Regression ---")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

# Linear Regression Evaluation with MSE and R²
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"MSE with Linear Regression : {mse_linear}")
print(f"R² with Linear Regression : {r2_linear}")

# === SIMPLE XGBOOST === #
print("\n--- XGBoost (simple) ---")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"MSE with XGBoost (simple) : {mse_xgb}")
print(f"R² with XGBoost (simple) : {r2_xgb}")

# === OPTIMISED XGBOOST === #
print("\n--- XGBoost (optimised) ---")
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)

best_params_xgb = grid_search_xgb.best_params_
print(f"Best hyperparameters for XGBoost : {best_params_xgb}")

best_xgb_model = grid_search_xgb.best_estimator_

y_pred_best_xgb = best_xgb_model.predict(X_test)

mse_best_xgb = mean_squared_error(y_test, y_pred_best_xgb)
r2_best_xgb = r2_score(y_test, y_pred_best_xgb)

print(f"MSE with the best XGBoost : {mse_best_xgb}")
print(f"R² with the best XGBoost : {r2_best_xgb}")

# === OUTCOME VISUALIZATION === #
plt.figure(figsize=(10, 6))

# Linear Regression
plt.scatter(y_test, y_pred_linear, alpha=0.7, label="Linear Regression", color="orange")

# Best XGBoost
plt.scatter(y_test, y_pred_best_xgb, alpha=0.7, label="XGBoost (Best)", color="blue")

# Ideal line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Idéal")

plt.title("Real Values vs Predictions")
plt.xlabel("Real Values")
plt.ylabel("Predictions")
plt.legend()
plt.show()
