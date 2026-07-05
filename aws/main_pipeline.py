import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier

from data_ingestion import ingest_data
from preprocessing import Preprocessing
from train_model import RandomForestModel, HistGradientBoostingModel, ExtraTreesModel
from evaluation import Evaluation

# MLFlow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
EVAL_DIR = BASE_DIR / "eval"


class CreditScoringPipeline:
    def __init__(self):
        # Pipeline PUNYA (HAS-A) preprocessing, daftar model, dan evaluator
        self.preprocessing = Preprocessing()
        self.evaluator = Evaluation()
        self.models = [
            RandomForestModel(),
            HistGradientBoostingModel(),
            ExtraTreesModel(),
        ]

    def run(self):
        # 1. Ingest data
        print("Step 1: Data Ingestion")
        ingest_data()
        df = pd.read_csv("ingested/data_C.csv")

        # 2. Cleaning
        print("Step 2: Cleaning")
        df = self.preprocessing.clean(df)

        # 3. Split (stratify karena target imbalanced)
        X = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 3b. Simpan train/test split (cleaned, SEBELUM encoding).
        # train.ipynb baca file ini buat refit preprocessing+model di bawah
        print("Step 2b: Simpan train/test split mentah")
        TRAIN_DIR.mkdir(exist_ok=True)
        TEST_DIR.mkdir(exist_ok=True)
        X_train.assign(Credit_Score=y_train.values).to_csv(TRAIN_DIR / "train.csv", index=False)
        X_test.assign(Credit_Score=y_test.values).to_csv(TEST_DIR / "test.csv", index=False)

        # 4. Preprocessing: fit di train, transform di test (no leakage)
        X_train_t = self.preprocessing.fit_transform(X_train)
        X_test_t = self.preprocessing.transform(X_test)

        mlflow.set_experiment("Credit_Score_Classification")

        # 5. Latih + evaluasi tiap model, catat ke MLflow
        print("\nStep 3: Training & Evaluation")
        results = []
        eval_report = {}
        for model in self.models:
            with mlflow.start_run(run_name=model.name):
                mlflow.log_param("model_type", model.name)
                model.train(X_train_t, y_train)
                metrics = self.evaluator.evaluate(model, X_test_t, y_test)
                mlflow.sklearn.log_model(model.model, artifact_path="model")
                results.append((model, metrics["f1_score"]))
                eval_report[model.name] = metrics

        # 6. Pilih model terbaik berdasarkan F1
        best_model, best_f1 = max(results, key=lambda r: r[1])
        print(f"\nModel terbaik: {best_model.name} (F1 = {best_f1:.4f})")

        # 7. Tuning Random Forest.
        #    Cari best params dengan preprocessing di DALAM CV
        print("\nStep 4: Hyperparameter Tuning Random Forest")
        tuning_pipe = SkPipeline(steps=[
            ('preprocessor', Preprocessing().build_preprocessor()),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [20, None],
            'classifier__min_samples_leaf': [6, 1]
        }
        grid = GridSearchCV(tuning_pipe, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Ambil best params
        best_params = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}
        print(f"Best CV F1 (rata-rata 3 lipatan validasi): {grid.best_score_:.4f}")

        tuned = RandomForestModel()
        tuned.name = "Random_Forest_Tuned"
        tuned.model.set_params(**best_params)
        tuned.train(X_train_t, y_train)

        with mlflow.start_run(run_name="Random_Forest_Tuned"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_f1", grid.best_score_)
            tuned_metrics = self.evaluator.evaluate(tuned, X_test_t, y_test)
            mlflow.sklearn.log_model(tuned.model, artifact_path="model")
            print(f"Best params: {best_params}")
            self.evaluator.report(tuned, X_test_t, y_test)
        eval_report["Random_Forest_Tuned"] = tuned_metrics
        eval_report["best_cv_f1"] = grid.best_score_

        # 8. Simpan artifact untuk deployment (local copy + bekal buat train.ipynb)
        ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump(self.preprocessing, ARTIFACT_DIR / "preprocessing.pkl", compress=9)
        joblib.dump(tuned.model, ARTIFACT_DIR / "best_model.pkl", compress=9)
        with open(ARTIFACT_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # 9. Simpan eval/metrics -> dipakai juga buat perbandingan local vs cloud
        EVAL_DIR.mkdir(exist_ok=True)
        with open(EVAL_DIR / "metrics.json", "w") as f:
            json.dump(eval_report, f, indent=2)

        print("\nPipeline selesai. Artifact: artifacts/ | train-test split: train/ test/ | eval: eval/")


if __name__ == "__main__":
    pipeline = CreditScoringPipeline()
    pipeline.run()
