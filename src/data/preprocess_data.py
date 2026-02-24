from datetime import datetime

import pandas as pd  # type: ignore

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _parse_record(line: str) -> dict:
    """
    Parse a single raw text line into a dictionary of key-value pairs.

    The input line is expected to contain fields separated by the '|'
    character, where each field has the format "key:value". Splitting
    occurs on the first colon only. The "ID" field is ignored.

    Args:
        line: A raw string from the text file (e.g., "key:value|key:value").

    Returns:
        A dictionary containing parsed keys and values with surrounding
        whitespace removed.
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


def _sanitize_sender(sender: str, sender_to_mask: str = "inny") -> str:
    """Normalize and optionally mask the sender value.

    If the sender is empty, missing, or matches the specified masking
    value (case-insensitive), a placeholder token is returned.

    Args:
        sender: Raw sender name extracted from the record.
        sender_to_mask: Sender value that should be replaced with a mask
            (default: "inny").

    Returns:
        A sanitized sender string or the "<MASK>" placeholder.
    """
    if not sender or sender.lower() == sender_to_mask:
        return "<MASK>"

    return sender


def _build_text_field(record: dict) -> str:
    """Construct a structured text representation from a parsed record.

    The function assembles selected fields into a single multi-line text
    block with explicit tags, suitable for downstream NLP processing.

    Included fields (if present):
        - Type → "[TYPE]"
        - Sender_brand → "[SENDER]" (sanitized)
        - Title → "[TITLE]"
        - Content → "[CONTENT]"

    Args:
        record: Dictionary containing parsed record fields.

    Returns:
        A formatted string combining available fields separated by
        newline characters.
    """
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


# Main:
def main() -> None:
    """
    Aggregate raw text datasets into a single processed dataset.

    This function scans model-specific subdirectories within the raw
    data directory, parses all .txt files, converts each non-empty line
    into a structured record, and appends metadata indicating the source
    model directory. A consolidated DataFrame is then saved in both
    Parquet and CSV formats with a timestamped filename.

    Workflow:
        1. Validate existence of the raw data directory.
        2. Iterate through each subdirectory representing a model source.
        3. Parse each non-empty line from .txt files into records.
        4. Build a structured text field for NLP usage.
        5. Aggregate all records into a DataFrame.
        6. Save outputs to the processed data directory.

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
    main()
