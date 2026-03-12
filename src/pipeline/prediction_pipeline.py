import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import joblib # pyright: ignore[reportMissingImports]

class PredictionPipeline:

    def predict(self,data):

        model=joblib.load("artifacts/model.pkl")

        preprocessor=joblib.load("artifacts/preprocessor.pkl")

        data_scaled=preprocessor.transform(data)

        prediction=model.predict(data_scaled)

        return prediction