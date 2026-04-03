import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

class DataPreprocessing:

    def initiate_preprocessing(self,train_path,test_path):

        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        X_train=train_df.drop("median_house_value",axis=1)
        y_train=train_df["median_house_value"]

        X_test=test_df.drop("median_house_value",axis=1)
        y_test=test_df["median_house_value"]

        # Select numeric and categorical columns
        num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # Create pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='mean')),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features)
        ])

        X_train=preprocessor.fit_transform(X_train)
        X_test=preprocessor.transform(X_test)

        os.makedirs("artifacts",exist_ok=True)

        joblib.dump(preprocessor,"artifacts/preprocessor.pkl")

        return X_train,X_test,y_train,y_test