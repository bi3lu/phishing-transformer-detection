"""Configuration module for phishing detection project.

Defines project paths and global constants used across the codebase.
"""

from pathlib import Path

# Paths:
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"  # TODO: Think about store data in separate place, like cloud or db

PROCESSED_DATA_DIR = DATA_DIR / "processed"  # TODO: Think about store data in separate place, like cloud or db

SPLIT_DATA_DIR = DATA_DIR / "split"  # TODO: Think about store data in separate place, like cloud or db

# Constants:
RANDOM_STATE = 42
LABEL_COL = "Is_Phishing"
TEXT_COL = "Text"
