import logging
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _parse_record(line: str) -> dict:
    """
    Parses a single line of text into a dictionary by splitting key-value pairs.

    The function splits the input string using the '|' separator and then divides
    each segment into a key and a value at the first occurrence of a colon.
    The 'ID' field is explicitly skipped as per requirements.

    Args:
        line (str): A raw string from the text file (expected format: key:value|key:value).

    Returns:
        dict: A dictionary of parsed data with keys and values stripped of
              leading/trailing whitespace.
    """
    parts = line.strip().split("|")
    record = {}

    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)

            if key.strip() == "ID":
                continue

            record[key.strip()] = value.strip()

    return record


def _sanitize_sender(
    sender: str, sender_to_mask: str = "inny"
) -> str:  # TODO: Add docstring
    if not sender or sender.lower() == sender_to_mask:
        return "<MASK>"

    return sender


def _build_text_field(record: dict) -> str:  # TODO: Add docstring
    parts = []

    if "Type" in record:
        parts.append(f"[TYPE] {record['Type']}")

    if "Sender_brand" in record:
        sender = _sanitize_sender(record.get("Sender_brand", "").strip())
        parts.append(f"[SENDER] {sender}")

    if "Title" in record:
        parts.append(f"[TITLE] {record['Title']}")

    if "Content" in record:
        parts.append(f"[CONTENT] {record['Content']}")

    return "\n".join(parts)


# Main processing function:
def preprocess_data() -> None:
    """
    Main orchestration function to merge sub-datasets into a single processed file.

    This function iterates through model-specific subdirectories in the raw data
    folder, parses all available text files, and appends the directory name
    as the 'Model_Source'. The final aggregated data is exported to both
    Parquet and CSV formats with timestamp.

    Workflow:
        1. Verifies if the raw data directory exists.
        2. Iterates through each model's subdirectory.
        3. Parses every non-empty line within the found .txt files.
        4. Aggregates the records into a pandas DataFrame.
        5. Ensures the output directory exists and saves the processed files with timestamp.

    Returns:
        None
    """
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
                            record["Text"] = _build_text_field(record)
                            all_data.append(record)

    if not all_data:
        logger.warning("No data found to process.")
        return

    df = pd.DataFrame(all_data)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get current timestamp:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to .parquet:
    output_path_parquet = PROCESSED_DATA_DIR / f"processed_data_{timestamp}.parquet"
    df.to_parquet(output_path_parquet, index=False)

    # Save to .csv:
    output_path_csv = PROCESSED_DATA_DIR / f"processed_data_{timestamp}.csv"
    df.to_csv(output_path_csv, index=False)

    logger.info(f"Data processing complete. Files saved to {PROCESSED_DATA_DIR}")


# Entry point:
if __name__ == "__main__":
    preprocess_data()
