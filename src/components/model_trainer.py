from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from src.utils import evaluate_model, save_object


class ModelTrainer:

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor()
        }

        # evaluate models
        report = evaluate_model(X_train, y_train, X_test, y_test, models)

        # find best model
        best_model_name = max(report, key=report.get) # type: ignore

        best_model = models[best_model_name]

        # train best model
        best_model.fit(X_train, y_train)

        # save model
        save_object("artifacts/model.pkl", best_model)

        return best_model, report, best_model_name