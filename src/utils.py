import os
import joblib
from sklearn.metrics import r2_score


def save_object(file_path, obj):

    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    joblib.dump(obj, file_path)


def evaluate_model(X_train, y_train, X_test, y_test, models):

    report = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)

        report[name] = score

    return report

    