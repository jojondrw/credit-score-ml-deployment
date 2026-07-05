import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


class Evaluation:
    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        roc_auc = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')

        # Catat hasil evaluasi ke MLflow (run dibuka di main_pipeline)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"{model.name} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        return {"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}

    def report(self, model, X_test, y_test):
        # Classification report untuk model terbaik
        preds = model.predict(X_test)
        print(classification_report(
            y_test, preds,
            labels=['Good', 'Standard', 'Poor'],
            target_names=['Good', 'Standard', 'Poor']))
