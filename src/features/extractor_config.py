from typing import List, Set

URGENCY_KEYWORD: List[str] = [
    "natychmiast",
    "pilnie",
    "bez opóźnienia",
    "wygasa",
    "teraz",
    "szybko",
    "nie czekaj",
    "ostatnia szansa",
    "pośpiesz się",
]
""""""

THREAT_KEYWORD: List[str] = [
    "zablokowane",
    "zablokuje",
    "wznowić",
    "przywrócić",
    "odblokować",
    "zmień hasło",
    "weryfikacja",
    "dezaktywacja",
    "zagrożenie",
    "niebezpieczeństwo",
]
""""""

VERIFICATION_KEYWORD: List[str] = ["potwierdź", "kod", "pin", "hasło", "tożsamość", "zaloguj", "login"]
""""""

LEGIT_DOMAINS: Set[str] = {
    "mbank.pl",
    "pkobp.pl",
    "ing.pl",
    "santander.pl",
    "aliorbank.pl",
    "poczta-polska.pl",
    "inpost.pl",
    "allegro.pl",
    "facebook.com",
    "google.com",
    "xtb.com",
    "olx.pl",
    "onet.pl",
    "x.com",
    "wp.pl",
}
""""""

ACTION_KEYWORDS: List[str] = [
    "kliknij",
    "sprawdź",
    "pobierz",
    "otwórz",
    "wejdź",
    "zaktualizuj",
    "wyślij",
    "dokonaj",
    "opłać",
]
""""""

FINANCIAL_KEYWORDS: List[str] = [
    "pieniądze",
    "środki",
    "przelew",
    "faktura",
    "płatność",
    "koszt",
    "portfel",
    "inwestycje",
    "kredyt",
    "debet",
]
""""""
