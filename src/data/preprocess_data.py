import hashlib
import json
import re
from typing import Any, Dict, Iterator

import pandas as pd

from src.config import BASE_DIR, LABEL_COL, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.hygiene import (
    assert_group_label_consistency,
    exact_duplicate_mask,
    remove_generator_url_wrappers,
)
from src.data.schema import validate_records
from src.data.template_audit import cluster_templates
from src.features.extractor import FEATURE_NAMES, PhishingFeatureExtractor
from src.utils.artifacts import file_sha256
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses raw phishing email data into structured format.

    Parses raw text records, retains their provenance, removes known generator
    artefacts, deduplicates exact records, and builds model input using only
    fields that are observable in a real message.
    """

    def __init__(self) -> None:
        """Initialize the deterministic preprocessor."""

    @staticmethod
    def parse_record(line: str) -> Dict[str, Any]:
        """Parse a pipe-delimited record into a dictionary.

        Splits a line by pipe characters and processes key:value pairs,
        retaining the source-local ID for provenance.

        Args:
            line: A pipe-delimited string containing record data.

        Returns:
            Dictionary with parsed key-value pairs.
        """
        parts = line.strip().split("|")
        record = {}

        for part in parts:
            if ":" not in part:
                raise ValueError(f"Malformed pipe-delimited record segment: {part!r}")

            if ":" in part:
                key, value = part.split(":", 1)

                normalized_key = "Source_Record_ID" if key.strip() == "ID" else key.strip()

                if normalized_key in record:
                    raise ValueError(f"Duplicate field: {normalized_key}")
                record[normalized_key] = value.strip()

        return record

    @staticmethod
    def iter_records(text: str) -> Iterator[tuple[int, int, Dict[str, Any]]]:
        """Frame records by ID markers, including joined lines and multiline content.

        Source_Line and Source_Offset preserve exact origin. A physical line is
        not assumed to be a record (Bielik part2 joins records 002 and 003).
        """
        starts = list(re.finditer(r"(?<!\S)ID:\s*[^|\s]+\s*\|", text))
        if not starts or text[: starts[0].start()].strip():
            raise ValueError("Raw file must start with an ID record marker")
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            record = DataPreprocessor.parse_record(text[match.start() : end])
            yield text.count("\n", 0, match.start()) + 1, match.start(), record

    def build_text_field(self, record: Dict[str, Any]) -> str:
        """Construct a structured text field from record components.

        Combines message type, e-mail title, and content. ``Sender_brand`` is
        deliberately excluded: it is a generator annotation, not a reliably
        observable sender identity, and ``Inny`` was strongly label-correlated.

        Args:
            record: Dictionary containing record fields.

        Returns:
            Formatted text field with tagged components.
        """
        parts = []

        if "Type" in record:
            parts.append(f"[TYPE] {record['Type']}")

        title = str(record.get("Title", "")).strip()

        if title and title.casefold() != "brak":
            parts.append(f"[TITLE] {title}")

        parts.append(f"[CONTENT] {record.get('Content', '')}")
        return "\n".join(parts)


# Main:
def main() -> None:
    """Aggregate raw text datasets into a single processed dataset.

    Iterates through raw data in deterministic order, parses records, retains
    provenance, removes generator URL wrappers and exact duplicates, assigns
    near-duplicate template groups, and saves the canonical dataset.
    """
    logger.info("Starting canonical data preprocessing...")

    all_data = []
    preprocessor = DataPreprocessor()

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {RAW_DATA_DIR}")

    source_manifest = json.loads((BASE_DIR / "docs" / "data_sources.json").read_text())

    for model_dir in sorted(RAW_DATA_DIR.iterdir()):
        if model_dir.is_dir():
            logger.info(f"Processing model directory: {model_dir.name}")

            for file_path in sorted(model_dir.glob("*.txt")):
                source_key = str(file_path.relative_to(RAW_DATA_DIR))
                source_info = source_manifest["files"].get(source_key)
                if source_info is None or source_info["sha256"] != file_sha256(file_path):
                    raise ValueError(f"Unregistered or changed raw source: {source_key}")
                raw_text = file_path.read_text(encoding="utf-8")
                for line_number, offset, record in preprocessor.iter_records(raw_text):
                    record["Model_Source"] = source_info["analysis_source"]
                    record["Source_Verification"] = source_info["status"]
                    label = record.get(LABEL_COL)

                    if label not in ("False", "True", "0", "1"):
                        raise ValueError(f"Invalid label at {file_path}:{line_number}")

                    record[LABEL_COL] = label in ("True", "1")
                    record["Source_File"] = file_path.name
                    record["Source_Line"] = line_number
                    record["Source_Offset"] = offset
                    record["Type"] = {"Email": "E-mail"}.get(record.get("Type", ""), record.get("Type", ""))
                    original_content = str(record.get("Content", ""))
                    record["Had_Generator_URL_Wrapper"] = "google.com/search?q=" in original_content.casefold()
                    record["Content"] = remove_generator_url_wrappers(original_content)
                    record["Title"] = remove_generator_url_wrappers(str(record.get("Title", "")))
                    record["Text"] = preprocessor.build_text_field(record)
                    all_data.append(record)

    if not all_data:
        raise ValueError("No raw data found")

    df = pd.DataFrame(all_data)

    required_columns = {"Type", "Title", "Content", LABEL_COL, "Model_Source"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing_columns)}")

    validate_records(df)
    assert_group_label_consistency(df)

    # Annotation differences do not make two observable messages independent:
    duplicate_key = ["Type", "Title", "Content"]
    duplicate_mask = exact_duplicate_mask(df, duplicate_key)
    duplicates_df = df.loc[duplicate_mask].copy()
    df = df.loc[~duplicate_mask].copy().reset_index(drop=True)

    df, similarity_audit = cluster_templates(df, BASE_DIR / "docs" / "template_overrides.json")
    df["Record_ID"] = df.apply(
        lambda row: hashlib.sha256(
            json.dumps([row["Type"], row["Title"], row["Content"]], ensure_ascii=False).encode()
        ).hexdigest()[:24],
        axis=1,
    )
    assert_group_label_consistency(df)

    # Heuristic features are saved for descriptive audits only.  They are not
    # serialized into Text and therefore cannot become transformer shortcuts.
    extractor = PhishingFeatureExtractor()
    feature_frame = pd.DataFrame(
        (df["Title"].where(df["Title"] != "Brak", "") + " " + df["Content"]).map(extractor.get_all_features).tolist()
    )

    for feature_name in FEATURE_NAMES:
        df[feature_name] = feature_frame[feature_name].astype(int)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = "canonical_v2"

    output_path_csv = PROCESSED_DATA_DIR / f"processed_data_{timestamp}.csv"
    payload = df.to_csv(index=False)

    if output_path_csv.exists() and output_path_csv.read_text() != payload:
        raise ValueError("Canonical v2 data changed. Archive data/v2 before rebuilding; old files are preserved.")

    output_path_csv.write_text(payload, encoding="utf-8")
    similarity_audit.to_csv(PROCESSED_DATA_DIR / "template_similarity_audit.csv", index=False)

    if not duplicates_df.empty:
        duplicates_path = PROCESSED_DATA_DIR / f"duplicates_removed_{timestamp}.csv"
        duplicates_df.to_csv(duplicates_path, index=False)
        logger.warning(f"Removed {len(duplicates_df)} exact duplicate records; audit: {duplicates_path}")

    review_columns = ["Record_ID", "Model_Source", "Type", "Content", LABEL_COL]
    review = df[review_columns].copy()
    review["reviewed_label"] = ""
    review["reviewer"] = ""
    review["reason"] = ""
    review["status"] = "pending_human_review"
    review.to_csv(PROCESSED_DATA_DIR / "label_review_queue.csv", index=False)
    source_counts = {str(key): int(value) for key, value in df["Model_Source"].value_counts().items()}
    audit = {
        "raw_records": len(all_data),
        "protocol": "group_refit_v2",
        "test_status": "exploratory_reanalysis_previously_inspected_corpus",
        "mixed_label_template_groups": int((df.groupby("Template_Group")[LABEL_COL].nunique() > 1).sum()),
        "largest_template_group": int(df.groupby("Template_Group").size().max()),
        "raw_source_manifest_sha256": file_sha256(BASE_DIR / "docs" / "data_sources.json"),
        "exact_duplicates_removed": len(duplicates_df),
        "canonical_records": len(df),
        "template_groups": int(df["Template_Group"].nunique()),
        "raw_generator_wrapper_records_before_cleaning": int(
            sum(bool(record.get("Had_Generator_URL_Wrapper")) for record in all_data)
        ),
        "canonical_generator_wrapper_records_before_cleaning": int(df["Had_Generator_URL_Wrapper"].sum()),
        "generator_wrappers_remaining_in_content": int(
            df["Content"].str.contains("google.com/search?q=", case=False, regex=False).sum()
        ),
        "source_counts": source_counts,
        "largest_source_fraction": max(source_counts.values()) / len(df),
        "model_input_fields": ["Type", "Title", "Content"],
        "descriptive_feature_columns": list(FEATURE_NAMES),
        "feature_tags_in_model_text": 0,
        "excluded_generator_annotations": [
            "Sender_brand",
            "Suspicion_level",
            "LLM_likeness",
            "Phishing_technique",
            "Model_Source",
        ],
    }

    with (PROCESSED_DATA_DIR / f"data_audit_{timestamp}.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    logger.info(
        f"Data processing complete. Saved {len(df)} unique records in "
        f"{df['Template_Group'].nunique()} template groups to {output_path_csv}"
    )


if __name__ == "__main__":
    main()
