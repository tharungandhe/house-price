from sklearn.metrics import r2_score
import joblib

class ModelEvaluation:

    def evaluate(self,X_test,y_test):

        model=joblib.load("artifacts/model.pkl")

        y_pred=model.predict(X_test)

        score=r2_score(y_test,y_pred)

        return score