# Karta zbioru syntetycznego

## Zakres i pochodzenie

Polskojęzyczne SMS-y, e-maile i powiadomienia wygenerowane przez LLM. Etykieta oznacza
przypisaną przez generator klasę wiadomości, nie niezależnie potwierdzony
incydent. Zbiór nie reprezentuje losowej próbki rzeczywistego ruchu.

Historyczna kolekcja zawierała 3983 rekordy. Wersja historyczna zawierała 1991
wiadomości po deduplikacji i 1455 grup; liczebności v2 są wyliczane przez
preprocessing w `data/v2/processed/data_audit_canonical_v2.json`.

`docs/data_sources.json` wiąże każdy plik surowy z SHA-256 i deklarowanym
źródłem. Nazwy źródeł nie dowodzą rzeczywistej wersji generatora. Plik
`gpt_5.1/gpt_5.2_part2.txt` ma sprzeczne oznaczenia; do czasu wyjaśnienia
przypisano go do `gpt_version_unverified`, osobnego holdoutu pliku o nieznanej
wersji. Nie należy interpretować go jako zweryfikowanego nowego generatora.

Nie posiadamy w repozytorium oryginalnych promptów, parametrów generowania,
identyfikatorów sesji ani niezależnego przeglądu etykiet. Pola `null` i status
`pending_human_review` oznaczają brak informacji, nie jej domyślne wartości.

## Aktualizacja danych 14.09.2026

Aktualnie jest 2295 rekordów w 16 plikach. Nowa dostawa dodała 812 rekordów,
a równocześnie zmieniono `gemini_3_part2.txt` (2500 → 1000 rekordów) i usunięto
`gemini_3_part3.txt` (1000 rekordów). Manifest zachowuje poprzedni hash zmienionego
pliku i dane usuniętego pliku; surowych plików nie odtwarzamy ani nie edytujemy.

Po deduplikacji pozostają 2234 wiadomości (61 duplikatów usunięto),
2139 operacyjnych grup podobieństwa. Podział: 1565/334/335. Największa rodzina
liczy 4 wiadomości. Dodano typ Notification. Parser rozpoznaje 12 rekordów
Bielika part2 mimo 11 fizycznych linii; pozycję zachowują Source_Line i Source_Offset.
Nowe nazwy generatorów w manifeście są deklaracjami według dostarczonych
katalogów, bez niezależnego potwierdzenia wersji.

## Reguły etykietowania i przegląd

- Zachowujemy oryginalne binarne etykiety generatora.
- Identyczne obserwowalne wiadomości z różnymi etykietami wymagają ręcznego
  rozstrzygnięcia; pipeline przerywa działanie.
- Podobne wiadomości z różnymi etykietami pozostają jedną rodziną; etykiet
  nie zmienia się dla uzyskania czystych grup lub lepszych wyników.
- Ręczny przegląd powinien obejmować próbkę warstwową według źródła i klasy,
  wszystkie sprzeczności identycznych rekordów oraz kandydatów z audytu
  podobieństwa. Zalecane dwie niezależne osoby, uzasadnienie etykiety,
  rozstrzygnięcie sporów i miara zgodności. Nie przedstawiać tego jako wykonanego.
- Dla rozstrzygnięć etykiet prowadzić tabelę: Record_ID, original_label,
  reviewed_label, reviewer, date, reason. Zachować surowe dane bez zmian.

## Grupy i przeciek

Normalizacja obejmuje URL-e, e-maile, liczby, polskie miesiące i identyfikatory.
Zbliżone teksty łączone są bez użycia etykiet przez podobieństwo TF-IDF
znakowych n-gramów (3–5), próg 0.92. Pary od 0.85 są eksportowane do audytu.
Są to operacyjne rodziny podobieństwa, a nie dowód całkowitego wyeliminowania
podobieństw semantycznych. Próg jest ustalony niezależnie od wyników modeli.

`docs/template_overrides.json` pozwala zapisać pary identyfikatorów z audytu
w `merge` lub `separate`. Zakazy separacji obowiązują też dla połączeń
przechodnich. Po przeglądzie zmiana grup wymaga nowej wersji danych i wyników.

## Udostępnianie i odtwarzanie

Dane/modelowe wyniki są ignorowane przez Git. Archiwum tworzy:

```bash
uv run --frozen python -m src.data.archive data/releases/synthetic-v2.tar.gz
uv run --frozen python -m src.data.archive data/releases/synthetic-v2.tar.gz --verify
```

Archiwum obejmuje dane surowe, v2, kartę i manifesty; towarzyszy mu manifest
sum kontrolnych. Publikacja/hosting nie zostały wykonane. Osoba odtwarzająca
badanie musi otrzymać archiwum wraz z manifestem od autora; sam klon kodu nie
zawiera zbioru. Nie przypisujemy danym automatycznie licencji kodu MIT.

## Ograniczenia i nowy test

Korpus był analizowany, a historyczny test obejrzany 15.07.2026. Nowe podziały
tego korpusu mają status **eksploracyjnej ponownej analizy**. Nie są nowym,
wcześniej niewidzianym testem. Do potwierdzenia wniosków trzeba pozyskać nowy
zbiór, sprawdzić grupy względem całego dotychczasowego korpusu, udokumentować
etykiety i zamknąć go przed ustaleniem ostatecznych modeli. Nie wygenerowano
fikcyjnego „niezależnego” testu z tych samych obserwacji.
