from pathlib import Path
import pandas as pd

# Base directory
BASE_DIR = Path(__file__).parent

# Tentukan folder data mentah dan folder hasil ingestion
RAW_DIR = BASE_DIR
INGESTED_DIR = BASE_DIR / "ingested"

# Sesuaikan nama file dengan dataset kamu
INPUT_FILE = RAW_DIR / "data_C.csv"
OUTPUT_FILE = INGESTED_DIR / "data_C.csv"

def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data
    df = pd.read_csv(INPUT_FILE)

    # Basic validation
    assert not df.empty, "Dataset is empty"

    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] Data ingested from {INPUT_FILE} -> {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()
