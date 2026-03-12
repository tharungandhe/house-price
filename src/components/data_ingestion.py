import os
import pandas as pd

class DataIngestion:

    def initiate_data_ingestion(self):

        df=pd.read_csv("houseing.csv")

        os.makedirs("artifacts",exist_ok=True)

        train_path="artifacts/train.csv"
        test_path="artifacts/test.csv"

        from sklearn.model_selection import train_test_split

        train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

        train_set.to_csv(train_path,index=False)
        test_set.to_csv(test_path,index=False)

        return train_path,test_path