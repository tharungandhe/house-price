import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

class DataPreprocessing:

    def initiate_preprocessing(self,train_path,test_path):

        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        X_train=train_df.drop("price",axis=1)
        y_train=train_df["price"]

        X_test=test_df.drop("price",axis=1)
        y_test=test_df["price"]

        num_features=X_train.columns

        scaler=StandardScaler()

        preprocessor=ColumnTransformer([
            ("scaler",scaler,num_features)
        ])

        X_train=preprocessor.fit_transform(X_train)
        X_test=preprocessor.transform(X_test)

        os.makedirs("artifacts",exist_ok=True)

        joblib.dump(preprocessor,"artifacts/preprocessor.pkl")

        return X_train,X_test,y_train,y_test