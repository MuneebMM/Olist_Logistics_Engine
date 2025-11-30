import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "processed" / "olist.db"
MODEL_PATH = BASE_DIR / "models" / "model_rf_v1.pkl"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
