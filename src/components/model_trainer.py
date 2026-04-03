from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.utils import save_object


class ModelTrainer:

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        base_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42)
        }

        param_grids = {
            "RandomForest": {
                "n_estimators": [50, 100],
                "max_depth": [10, None]
            }
        }

        tuned_models = {}
        tuning_info = {}

        for name, model in base_models.items():
            if name in param_grids:
                print(f"Tuning {name} with GridSearchCV...")
                grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                tuned_models[name] = grid_search.best_estimator_
                tuning_info[name] = {
                    "best_params": grid_search.best_params_,
                    "best_score": grid_search.best_score_
                }
                print(f"Best params for {name}: {grid_search.best_params_}")
            else:
                print(f"Training {name} with default parameters...")
                model.fit(X_train, y_train)
                tuned_models[name] = model
                tuning_info[name] = "default parameters"

        report = {}
        for name, model in tuned_models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            report[name] = {
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "mape": mape
            }

        # choose best model by R2
        best_model_name = max(report, key=lambda nm: report[nm]["r2"])
        best_model = tuned_models[best_model_name]

        # save chosen model and tuning metadata
        save_object("artifacts/model.pkl", best_model)
        save_object("artifacts/model_tuning_info.pkl", tuning_info)
        save_object("artifacts/model_evaluation_report.pkl", report)

        return best_model, report, best_model_name