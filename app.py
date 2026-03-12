from flask import Flask,request,render_template # type: ignore
import pandas as pd  # type: ignore
from src.pipeline.prediction_pipeline import PredictionPipeline

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    area=float(request.form["area"])
    bedrooms=float(request.form["bedrooms"])
    bathrooms=float(request.form["bathrooms"])

    data=pd.DataFrame({
        "area":[area],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms]
    })

    pipeline=PredictionPipeline()

    result=pipeline.predict(data)

    return render_template("index.html",prediction_text=f"House Price is {result[0]}")

if __name__=="__main__":
    app.run(debug=True)