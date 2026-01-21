from pathlib import Path

BASE_DATA_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DATA_DIR = BASE_DATA_DIR / "processed"


if __name__ == "__main__":
    print("Hello world!")
