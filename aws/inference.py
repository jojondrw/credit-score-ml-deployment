import joblib
import pandas as pd
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class CreditScoringInference:
    def __init__(self, artifact_dir=None):
        artifact_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
        self.preprocessing = joblib.load(artifact_dir / "preprocessing.pkl")
        self.model = joblib.load(artifact_dir / "best_model.pkl")

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

# load model
def model_fn(model_dir):
    return CreditScoringInference(artifact_dir=model_dir)

# parse request
def input_fn(request_body, request_content_type):
    payload = json.loads(request_body)
    return payload["instances"]  # list of dict, satu dict = satu nasabah

# run prediction
def predict_fn(instances, infer_obj):
    return [infer_obj.predict(row) for row in instances]

# format response
def output_fn(prediction, accept_content_type):
    return json.dumps(prediction), accept_content_type