import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("house/1553768847-housing.csv")

# Handle missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Features & target (Drop ocean_proximity to match streamlit input)
X = df.drop(["median_house_value", "ocean_proximity"], axis=1)
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n--- Linear Regression Performance ---")
print(f"Accuracy (R² Score): {r2_score(y_test, lr_pred) * 100:.2f}%")
print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mean_squared_error(y_test, lr_pred)):,.2f}")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, lr_pred):,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, lr_pred) * 100:.2f}%")
print("-------------------------------------\n")

# Train Random Forest
print("Training Random Forest... (this might take a few seconds)")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print(f"Accuracy (R² Score): {r2_score(y_test, rf_pred) * 100:.2f}%")
print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mean_squared_error(y_test, rf_pred)):,.2f}")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, rf_pred):,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, rf_pred) * 100:.2f}%")
print("-------------------------------------\n")

# ✅ SAVE AFTER TRAINING (IMPORTANT)
joblib.dump(rf_model, "model.pkl")

print("Random Forest (Best Model) trained and saved!")