from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
import os
import joblib

class TrainPipeline:

    def start_training(self):
        model_path = "artifacts/model.pkl"
        report_path = "artifacts/model_evaluation_report.pkl"
        
        if os.path.exists(model_path) and os.path.exists(report_path):
            print("Loading existing trained model and evaluation report...")
            best_model = joblib.load(model_path)
            report = joblib.load(report_path)
            best_model_name = max(report, key=lambda nm: report[nm]["r2"])
            return best_model, report, best_model_name
        elif os.path.exists(model_path):
            print("Model exists but no evaluation report found. Retraining to get metrics...")
            # Fall through to training
        else:
            print("No existing model found. Starting training...")

        ingestion=DataIngestion()

        train_path,test_path=ingestion.initiate_data_ingestion()

        preprocess=DataPreprocessing()

        X_train,X_test,y_train,y_test=preprocess.initiate_preprocessing(train_path,test_path)

        trainer=ModelTrainer()

        best_model, report, best_model_name=trainer.initiate_model_trainer(X_train,X_test,y_train,y_test)
        
        return best_model, report, best_model_name