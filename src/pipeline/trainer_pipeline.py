from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def start_training(self):

        ingestion=DataIngestion()

        train_path,test_path=ingestion.initiate_data_ingestion()

        preprocess=DataPreprocessing()

        X_train,X_test,y_train,y_test=preprocess.initiate_preprocessing(train_path,test_path)

        trainer=ModelTrainer()

        best_model, report, best_model_name=trainer.initiate_model_trainer(X_train,X_test,y_train,y_test)
        
        return best_model, report, best_model_name