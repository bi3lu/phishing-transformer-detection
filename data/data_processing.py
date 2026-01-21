import logging
from pathlib import Path

import colorlog  # type: ignore
import pandas as pd  # type: ignore

BASE_DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DATA_DIR / "raw"
MERGED_DATA_DIR = BASE_DATA_DIR / "merged"

# Setup logging:
logger = logging.getLogger(__name__)

if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.DEBUG)

log_format = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)

logger.addHandler(console_handler)


# Helper functions:
def _parse_record(line: str) -> dict:  # TODO: Add docstring
    parts = line.strip().split("|")
    record = {}

    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)

            if key.strip() == "ID":
                continue

            record[key.strip()] = value.strip()

    return record


# Main processing function:
def process_data() -> None:  # TODO: Add docstring
    logger.info("Starting data processing...")
    
    all_data = []

    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory does not exist: {RAW_DATA_DIR}")
        return

    for model_dir in RAW_DATA_DIR.iterdir():
        if model_dir.is_dir():
            logger.info(f"Processing model directory: {model_dir.name}")

            for file_path in model_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            record = _parse_record(line)
                            record["Model_Source"] = model_dir.name
                            all_data.append(record)

    if not all_data:
        logger.warning("No data found to process.")
        return

    df = pd.DataFrame(all_data)

    MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to .parquet:
    output_path_parquet = MERGED_DATA_DIR / "merged_raw_data.parquet"
    df.to_parquet(output_path_parquet, index=False)

    # Save to .csv:
    output_path_csv = MERGED_DATA_DIR / "merged_raw_data.csv"
    df.to_csv(output_path_csv, index=False)

    logger.info(f"Data processing complete. Files saved to {MERGED_DATA_DIR}")


# Entry point:
if __name__ == "__main__":
    process_data()
