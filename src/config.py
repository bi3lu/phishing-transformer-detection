"""Versioned research paths. Legacy data and results are never overwritten."""

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "group_refit_v2"
RUN_ID = os.environ.get("PHISHING_RUN_ID", "protocol_v2")

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", RUN_ID):
    raise ValueError("PHISHING_RUN_ID must be a simple directory name")

RANDOM_STATE = int(os.environ.get("PHISHING_SEED", "42"))
SPLIT_RANDOM_STATE = 42  # Fixed across training seeds to isolate optimization variance.
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "v2" / "processed"
SPLIT_DATA_DIR = DATA_DIR / "v2" / "split"
RESULTS_DIR = BASE_DIR / "results" / "runs" / RUN_ID / f"seed-{RANDOM_STATE}"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"
LABEL_COL = "Is_Phishing"
TEXT_COL = "Text"
CONTENT_COL = "Content"
SOURCE_COL = "Model_Source"
TEMPLATE_GROUP_COL = "Template_Group"
DEFAULT_MAX_LENGTH = 256
