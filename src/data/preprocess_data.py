from datetime import datetime
from typing import Any, Dict

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


class DataPreprocessor:
    """Encapsulates logic for parsing and structuring raw phishing data."""

    @staticmethod
    def parse_record(line: str) -> Dict[str, Any]:
        """Parse a single raw text line into a dictionary of key-value pairs.

        The input line is expected to contain fields separated by the '|'
        character, where each field has the format "key:value". Splitting
        occurs on the first colon only. The "ID" field is ignored.

        Args:
            line: A raw string from the text file.

        Returns:
            A dictionary containing parsed keys and values.
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

    @staticmethod
    def sanitize_sender(sender: str, sender_to_mask: str = "inny") -> str:
        """Normalize and optionally mask the sender value.

        Args:
            sender: Raw sender name.
            sender_to_mask: Value to replace with mask (default: "inny").

        Returns:
            Sanitized sender or "<MASK>".
        """
        if not sender or sender.lower() == sender_to_mask:
            return "<MASK>"
        return sender

    @classmethod
    def build_text_field(cls, record: Dict[str, Any]) -> str:
        """Construct a structured text representation from a parsed record.

        Args:
            record: Dictionary containing parsed record fields.

        Returns:
            Formatted string for NLP processing.
        """
        parts = []

        if "Type" in record:
            parts.append(f"[TYPE] {record['Type']}")

        if "Sender_brand" in record:
            sender = cls.sanitize_sender(record.get("Sender_brand", "").strip())
            parts.append(f"[SENDER] {sender}")

        if "Title" in record:
            parts.append(f"[TITLE] {record['Title']}")

        if "Content" in record:
            parts.append(f"[CONTENT] {record['Content']}")

        return "\n".join(parts)


# Main:
def main() -> None:
    """Aggregate raw text datasets into a single processed dataset."""
    logger.info("Starting data processing...")

    all_data = []
    preprocessor = DataPreprocessor()

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
                            record = preprocessor.parse_record(line)
                            record["Model_Source"] = model_dir.name
                            record["Text"] = preprocessor.build_text_field(record)
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
