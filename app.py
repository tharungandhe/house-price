from flask import Flask,request,render_template 
import pandas as pd  
from src.pipeline.prediction_pipeline import PredictionPipeline
import os

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    longitude=float(request.form["longitude"])
    latitude=float(request.form["latitude"])
    housing_median_age=float(request.form["housing_median_age"])
    total_rooms=float(request.form["total_rooms"])
    total_bedrooms=float(request.form["total_bedrooms"])
    population=float(request.form["population"])
    households=float(request.form["households"])
    median_income=float(request.form["median_income"])

    data=pd.DataFrame({
        "longitude":[longitude],
        "latitude":[latitude],
        "housing_median_age":[housing_median_age],
        "total_rooms":[total_rooms],
        "total_bedrooms":[total_bedrooms],
        "population":[population],
        "households":[households],
        "median_income":[median_income]
    })

    pipeline=PredictionPipeline()

    result=pipeline.predict(data)

    return render_template("index.html",prediction_text=f"House Price is ${result[0]:,.2f}")

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)