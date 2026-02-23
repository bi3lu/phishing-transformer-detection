from pathlib import Path

# Paths:
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "split"

# Constants:
RANDOM_STATE = 42
LABEL_COL = "Is_Phishing"
TEXT_COL = "Text"
