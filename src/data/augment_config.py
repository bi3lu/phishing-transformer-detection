"""Configuration data for text augmentation strategies.

Defines dictionaries and lists of text patterns used for data augmentation
in phishing detection, including character substitutions and brand keywords.
"""

from typing import Dict, List

SHORTCUTS: List[str] = [
    "InPost",
    "DPD",
    "mBank",
    "Allegro",
    "Netflix",
    "Poczta Polska",
    "ING",
    "PKO BP",
    "Pekao",
    "InPost",
    "OLX",
    "PayPal",
    "XTB",
    "ZUS",
    "ePUAP",
    "PZU",
    "Media Expert",
    "RTV Euro AGD",
    "morele.net",
    "DHL",
    "T-Mobile",
    "US",
    "Orange",
    "Empik",
    "Millennium",
    "Trading 212",
    "Inny",  # NOTE: now for test...
]
""""""

HOMOGLYPHS: Dict[str, List[str]] = {
    "a": ["а", "@"],
    "o": ["0", "о"],
    "e": ["е", "3"],
    "i": ["1", "l", "!"],
    "s": ["5", "$"],
    "p": ["р"],
    "l": ["1", "i"],
}
""""""
