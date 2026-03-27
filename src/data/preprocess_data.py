from datetime import datetime
from typing import Any, Dict

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features.extractor import PhishingFeatureExtractor
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses raw phishing email data into structured format.

    Parses raw text records, sanitizes sender information, and builds
    structured text fields for model training.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with a feature extractor."""
        self.extractor = PhishingFeatureExtractor()

    @staticmethod
    def parse_record(line: str) -> Dict[str, Any]:
        """Parse a pipe-delimited record into a dictionary.

        Splits a line by pipe characters and processes key:value pairs,
        skipping the ID field.

        Args:
            line: A pipe-delimited string containing record data.

        Returns:
            Dictionary with parsed key-value pairs.
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
        """Sanitize sender information by masking generic entries.

        Replaces empty or generic sender values with a mask token.

        Args:
            sender: The sender name to sanitize.
            sender_to_mask: Value to mask (case-insensitive). Defaults to 'inny'.

        Returns:
            Masked sender ('<MASK>') or original sender name.
        """
        if not sender or sender.lower() == sender_to_mask:
            return "<MASK>"

        return sender

    def build_text_field(self, record: Dict[str, Any]) -> str:
        """Construct a structured text field from record components.

        Combines type, sender, and content into a formatted text field
        with semantic tags for model input.

        Args:
            record: Dictionary containing record fields.

        Returns:
            Formatted text field with tagged components.
        """
        parts = []

        if "Type" in record:
            parts.append(f"[TYPE] {record['Type']}")

        if "Sender_brand" in record:
            sender = self.sanitize_sender(record.get("Sender_brand", "").strip())
            parts.append(f"[SENDER] {sender}")

        parts.append(f"[CONTENT] {record.get('Content', '')}")
        return "\n".join(parts)


# Main:
def main() -> None:
    """Aggregate raw text datasets into a single processed dataset.

    Iterates through all raw data directories, parses records, sanitizes
    sender information, constructs text fields, and saves the aggregated
    dataset as a timestamped CSV file in the processed data directory.
    """
    logger.info("Starting data processing with Feature Injection...")

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

    # Normalize Type column ("Email" -> "E-mail"):
    if "Type" in df.columns:
        df["Type"] = df["Type"].replace({"Email": "E-mail"})

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path_csv = PROCESSED_DATA_DIR / f"processed_data_{timestamp}.csv"
    df.to_csv(output_path_csv, index=False)

    logger.info(f"Data processing complete. Saved {len(df)} records to {output_path_csv}")


if __name__ == "__main__":
    main()
