import pandas as pd
import joblib

# Load trained model (make sure you already ran main.py)
model = joblib.load("model.pkl")

print("="*60)
print("🏠 House Price Prediction System")
print("="*60)

# Take user inputs
longitude = float(input("Enter Longitude: "))
latitude = float(input("Enter Latitude: "))
housing_median_age = int(input("Enter House Age: "))
total_rooms = int(input("Enter Total Rooms: "))
total_bedrooms = int(input("Enter Total Bedrooms: "))
population = int(input("Enter Population: "))
households = int(input("Enter Households: "))
median_income = float(input("Enter Median Income: "))

print("\nChoose Ocean Proximity:")
print("1. NEAR BAY")
print("2. INLAND")
print("3. NEAR OCEAN")
print("4. ISLAND")
print("5. <1H OCEAN")

choice = int(input("Enter choice (1-5): "))

mapping = {
    1: "NEAR BAY",
    2: "INLAND",
    3: "NEAR OCEAN",
    4: "ISLAND",
    5: "<1H OCEAN"
}

ocean_proximity = mapping.get(choice, "INLAND")

# Create DataFrame
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# Predict
prediction = model.predict(input_data)

print("\n" + "="*60)
print(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
print("="*60)
    # Generate HTML report
  