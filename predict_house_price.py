import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline
import warnings
warnings.filterwarnings('ignore')

# Input data for prediction using actual dataset columns
# The dataset contains: longitude, latitude, housing_median_age, total_rooms, 
# total_bedrooms, population, households, median_income, ocean_proximity

longitude = -122.25  # Medapally area coordinates (example)
latitude = 37.85
housing_median_age = 25
total_rooms = 1500
total_bedrooms = 4
population = 1200
households = 400
median_income = 5.5

print("=" * 60)
print("House Price Prediction - Medapally")
print("=" * 60)
print(f"Longitude: {longitude}")
print(f"Latitude: {latitude}")
print(f"Housing Median Age: {housing_median_age} years")
print(f"Total Rooms: {total_rooms}")
print(f"Total Bedrooms: {total_bedrooms}")
print(f"Population: {population}")
print(f"Households: {households}")
print(f"Median Income: ${median_income}0,000")
print("=" * 60)

data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income]
})

try:
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(data)
    
    predicted_price = prediction[0]
    
    print(f"\n✓ PREDICTED HOUSE PRICE: ${predicted_price:,.2f}")
    print("=" * 60)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>House Price Prediction - Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 90%;
            }}
            h1 {{
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }}
            .input-section {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .input-section h2 {{
                color: #666;
                margin-top: 0;
            }}
            .input-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }}
            .input-item {{
                padding: 10px;
                background: white;
                border-radius: 3px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .label {{
                font-weight: bold;
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 5px;
            }}
            .value {{
                color: #333;
                font-size: 16px;
            }}
            .result-section {{
                text-align: center;
                padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 5px;
                color: white;
            }}
            .result-label {{
                font-size: 18px;
                margin-bottom: 10px;
                opacity: 0.9;
            }}
            .result-price {{
                font-size: 48px;
                font-weight: bold;
                margin: 20px 0;
            }}
            .success {{
                color: #4CAF50;
                font-weight: bold;
                background: rgba(0,0,0,0.1);
                padding: 10px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>House Price Prediction</h1>
            
            <div class="input-section">
                <h2>Property Details - Medapally</h2>
                <div class="input-grid">
                    <div class="input-item">
                        <div class="label">Longitude</div>
                        <div class="value">{longitude}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Latitude</div>
                        <div class="value">{latitude}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Housing Age</div>
                        <div class="value">{housing_median_age} years</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Total Rooms</div>
                        <div class="value">{total_rooms}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Bedrooms</div>
                        <div class="value">{total_bedrooms}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Population</div>
                        <div class="value">{population}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Households</div>
                        <div class="value">{households}</div>
                    </div>
                    <div class="input-item">
                        <div class="label">Median Income</div>
                        <div class="value">${median_income}0,000</div>
                    </div>
                </div>
            </div>
            
            <div class="result-section">
                <div class="result-label">Predicted House Price</div>
                <div class="result-price">${predicted_price:,.2f}</div>
                <div class="success">[PREDICTED] Prediction Successful</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save and open HTML
    import webbrowser
    import os
    
    html_path = "artifacts/prediction_result.html"
    os.makedirs("artifacts", exist_ok=True)
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"\nOpening results in browser... ({html_path})")
    webbrowser.open("file://" + os.path.abspath(html_path))
    
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    print("Make sure to run main.py first to train the model.")
