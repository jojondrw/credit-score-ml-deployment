import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class CreditScoringInference:
    def __init__(self):
        # Load preprocessing + model yang sudah dilatih
        self.preprocessing = joblib.load(ARTIFACT_DIR / "preprocessing.pkl")
        self.model = joblib.load(ARTIFACT_DIR / "best_model.pkl")

    def predict(self, input_dict):
        # Bungkus input 1 nasabah jadi DataFrame 1 baris
        df = pd.DataFrame([input_dict])

        # Transform pakai preprocessing yang sama dengan training
        X = self.preprocessing.transform(df)

        # Prediksi label + probabilitas
        label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # Pasangkan tiap kelas dengan probabilitasnya
        proba_dict = {cls: float(p) for cls, p in zip(self.model.classes_, proba)}

        return {"prediction": label, "probabilities": proba_dict}