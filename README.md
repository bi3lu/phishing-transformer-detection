# Wykrywanie phishingu w języku polskim z wykorzystaniem transformerów i danych syntetycznych

Repozytorium zawiera implementację badania realizowanego w ramach pracy magisterskiej.
Celem jest ocena modeli transformerowych w binarnej klasyfikacji polskojęzycznych
wiadomości syntetycznych jako phishing lub wiadomość prawidłowa, ze szczególnym
uwzględnieniem generalizacji między źródłami generowania danych.

Badanie porównuje trzy modele transformerowe z klasyfikatorem referencyjnym opartym
na TF-IDF i regresji logistycznej. Obejmuje przygotowanie danych, kontrolę podobieństwa
szablonów, trening, walidację krzyżową, kalibrację prawdopodobieństw, analizę błędów
oraz interpretację predykcji za pomocą SHAP.

**Zakres wnioskowania:** protokół `group_refit_v2` służy eksploracyjnej reanalizie
korpusu, który był już wcześniej analizowany. Wyniki nie stanowią potwierdzenia
skuteczności w rzeczywistym ruchu ani oceny na nowym, niezależnym zbiorze testowym.

## Cel i pytania badawcze

1. Jakie wyniki uzyskują transformery w porównaniu z modelem referencyjnym?
2. Jak zmienia się skuteczność po wyłączeniu z treningu całego źródła danych syntetycznych?
3. Czy wysoki wynik łączny odpowiada równomiernej skuteczności dla poszczególnych źródeł?
4. Jak kalibracja, próg decyzyjny i łączenie modeli wpływają na ocenę końcową?
5. Jakie rodzaje błędów popełniają modele i które fragmenty wiadomości wpływają na predykcje?

## Modele

| Model | Identyfikator lub reprezentacja |
|---|---|
| Baseline | TF-IDF, n-gramy 1–3, regresja logistyczna z wagami klas |
| HerBERT | `allegro/herbert-base-cased` |
| Polish RoBERTa v2 | `sdadas/polish-roberta-base-v2` |
| Multilingual DistilBERT | `distilbert-base-multilingual-cased` |

Rewizje checkpointów i hiperparametry są zapisane w [params.yaml](params.yaml).
Modele otrzymują typ wiadomości, tytuł i treść. Etykieta, nazwa generatora oraz jego
adnotacje pomocnicze nie są cechami wejściowymi. Augmentacja jest wyłączona
w protokole podstawowym.

## Zbiór danych

Korpus obejmuje syntetyczne SMS-y, e-maile i powiadomienia w języku polskim.
Stan wersji danych przygotowanej 14 września 2026 r.:

| Właściwość | Liczebność |
|---|---:|
| Pliki źródłowe | 16 |
| Rekordy przed deduplikacją | 2295 |
| Unikalne wiadomości | 2234 |
| Operacyjne grupy podobieństwa | 2139 |
| Deklarowane źródła danych | 10 |
| Zbiór treningowy / walidacyjny / testowy | 1565 / 334 / 335 |

Preprocessing waliduje strukturę rekordów, zachowuje ich pochodzenie i usuwa
duplikaty. Rodziny podobnych wiadomości wyznaczane są bez użycia etykiet,
z uwzględnieniem normalizacji oraz podobieństwa znakowych n-gramów TF-IDF.
Podział treningowy, walidacyjny i testowy jest rozłączny względem tych rodzin.
Grupowanie ogranicza przeciek szablonów, ale nie dowodzi braku wszelkich
podobieństw semantycznych między zbiorami.

Szczegóły opisuje [karta zbioru](docs/dataset_card.md).
[Manifest źródeł](docs/data_sources.json) zawiera sumy SHA-256 plików,
a [reguły ręcznego grupowania](docs/template_overrides.json) umożliwiają jawne
zapisanie rozstrzygnięć audytu podobieństwa.

Dane surowe, modele i pełne wyniki są wyłączone ze śledzenia w Git. Odtworzenie
badania wymaga danych zgodnych z manifestem lub archiwum przekazanego przez autora.
Sam klon repozytorium nie zawiera korpusu.

## Protokół eksperymentalny

### Trening i ocena końcowa

1. Wewnętrzny podział grupowy w zbiorze treningowym służy wyborowi liczby epok.
2. Model jest ponownie inicjalizowany z checkpointu bazowego i trenowany na całym
   zbiorze treningowym przez wybraną liczbę epok.
3. Na zewnętrznym zbiorze walidacyjnym wykonywana jest kalibracja typu sigmoid
   na logitach prawdopodobieństw. Predykcje do wyboru progu pochodzą z pięciu
   rozłącznych grupowo foldów kalibracji.
4. Próg oraz skład ensemble są wybierane na walidacji. Ensemble uśrednia
   prawdopodobieństwa składowych z jednakowymi wagami.
5. Zamrożona konfiguracja jest oceniana na zbiorze testowym. Test nie jest
   wykorzystywany do wyboru epoki, kalibracji, progu ani składu ensemble.

### Walidacja krzyżowa

Oba protokoły CV wykorzystują wyłącznie połączone zbiory treningowy i walidacyjny:

- **Template-group CV:** pięć foldów rozłącznych względem rodzin szablonów,
  wspólnych dla porównywanych modeli.
- **Source-holdout:** dziesięć prób typu leave-one-source-out. Każde źródło jest
  kolejno wyłączane z treningu i wykorzystywane do oceny. Z treningu usuwane są
  także wiadomości należące do rodzin szablonów obecnych w wyłączonym źródle.

W każdym foldzie transformera liczba epok jest wybierana wewnątrz części
treningowej, po czym wykonywany jest ponowny trening na całej tej części.
Baseline również wykorzystuje całą dostępną część treningową.

**CV ocenia surowe modele przy progu 0,5.** Nie jest oceną całego procesu
kalibracji ani ensemble. Wyniki CV i końcowej ewaluacji opisują zatem różne
procedury predykcyjne.

### Miary i niepewność

Podstawową miarą jest F1 klasy phishing. Raportowane są również precision,
recall, accuracy, ROC-AUC, liczby FP/FN, Brier score, ECE i wyniki w podgrupach.

Przedziały ufności są wyznaczane przez bootstrap rodzin szablonów: 2000
resamplowań dla CV i 5000 dla testu. Opisują niepewność warunkową względem
wytrenowanych modeli; nie obejmują zmienności między seedami. Osobna miara
`group_all_correct` oznacza poprawną klasyfikację wszystkich wariantów rodziny.
Test McNemara z korekcją Holma porównuje modele na tej jednostce grupowej.

W source-holdout należy rozróżniać F1 obliczone ze wszystkich predykcji łącznie
od średniej F1 po źródłach, w której każde źródło ma jednakową wagę.
Nie są to zamienne miary.

## Wyniki: seed 42

Zakończono przebieg `thesis_v2_macos27/seed-42`, obejmujący 60 foldów CV
oraz ocenę pięciu konfiguracji na 335 wiadomościach testowych.
Poniżej podano F1 klasy phishing w procentach.

| Model | Template-group CV, łącznie | Source-holdout, łącznie | Średnia po źródłach | Test końcowy |
|---|---:|---:|---:|---:|
| Baseline | 95,97 | 88,47 | 90,89 | 95,31 |
| HerBERT | 99,30 | 98,03 | 90,54 | 98,12 |
| Polish RoBERTa v2 | 99,06 | 98,18 | 91,11 | 98,13 |
| Multilingual DistilBERT | 98,88 | 97,38 | 90,58 | 97,61 |
| Ensemble | — | — | — | 98,13 |

Zestawienie wartości niezaokrąglonych, przedziałów ufności dla testu i sum
kontrolnych raportów znajduje się w [podsumowaniu wyników](docs/results_seed42.json).

Transformery uzyskały wyższe zbiorcze F1 w source-holdout niż baseline, jednak
przy jednakowej wadze źródeł przewaga ta prawie zanika. Na teście zastosowany
test McNemara po korekcji Holma nie wykazał istotnych różnic między parami
modeli przy poziomie 0,05. Nie jest to dowód równoważności modeli.

Wyniki dotyczą jednego seeda. Zaplanowano także seedy 43 i 44, przy niezmienionym
podziale danych. Nie należy interpretować obecnego zestawienia jako zakończonej
oceny zmienności treningu.

## Odtworzenie eksperymentu

### Środowisko

Wymagany jest Python 3.10+ oraz `uv`. Lokalne eksperymenty wykonano na Pythonie
3.12.0, PyTorch 2.10.0 i Transformers 5.2.0. Zależności są zapisane w `uv.lock`.
Launcher wykorzystuje `fcntl`, dlatego jego blokady procesów wymagają systemu
Unix, np. macOS lub Linux.

```bash
uv sync --frozen --all-extras
```

Profil sprzętowy przygotowano dla MacBooka Pro M4 Max z 48 GB pamięci zunifikowanej:
MPS, float32, maksymalnie 256 tokenów, mikro-batch 8, akumulacja gradientów 2,
dynamiczny padding i batch ewaluacyjny 16. Modele są uruchamiane sekwencyjnie.
Limit MPS wynosi 80% pamięci rekomendowanej przez Metal; nie jest limitem całej
pamięci systemowej. Przy braku akceleratora kod może korzystać z CPU.

### Pełny przebieg dla jednego seeda

Polecenia należy wykonywać z katalogu głównego repozytorium. Nazwa przebiegu
identyfikuje konkretny kod, parametry i środowisko.

```bash
export PHISHING_RUN_ID=thesis_v2_macos27
export PHISHING_SEED=42
export TOKENIZERS_PARALLELISM=false

uv run --frozen python main.py
```

Na macOS można użyć `caffeinate -i uv run --frozen python main.py`, aby zapobiec
uśpieniu podczas obliczeń. Pierwsze uruchomienie wymaga pobrania checkpointów
z Hugging Face. Po ich pobraniu `HF_HUB_OFFLINE=1` pozwala korzystać z lokalnego
cache, również przy ograniczonym dostępie do sieci.

### Wybrane etapy i wiele seedów

```bash
# Sprawdzenie identyfikacji eksperymentu bez treningu i zapisu wyników:
uv run --frozen python main.py --check-run

# Przygotowanie danych:
uv run --frozen python main.py --only preprocess split

# Trening wszystkich modeli:
uv run --frozen python main.py --only baseline finetune

# Oba rodzaje CV:
uv run --frozen python main.py --only kfold source-holdout

# Kalibracja, wybór ensemble, ocena końcowa i analiza:
uv run --frozen python main.py --only threshold ensemble evaluate analysis

# Kontrola kompletności całego przebiegu:
uv run --frozen python main.py --only report

# Pełne badanie dla seedów 42, 43 i 44, sekwencyjnie:
bash scripts/run_study.sh
```

`--only` zachowuje kolejność etapów określoną w protokole. `--skip preprocess split`
pozwala wykorzystać przygotowane dane. Opcja `--experiments herbert-base` zawęża
trening lub CV, natomiast finalna ewaluacja wymaga wszystkich zarejestrowanych
modeli i baseline'u.

### Wznowienie i integralność

Powtórzenie polecenia z tą samą nazwą przebiegu i seedem wykorzystuje poprawne
modele i ukończone foldy. Przerwany fold rozpoczyna się od początku. Sumy
kontrolne wiążą wyniki z fizycznymi plikami danych i modeli.

Każdy model i protokół CV uruchamiany jest w osobnym procesie; osobne procesy
obsługują również etapy ewaluacji. Ogranicza to narastanie zasobów Metal między
treningami. OOM w CV jest automatycznie ponawiany w świeżym procesie wyłącznie
wtedy, gdy poprzednia próba ukończyła nowe foldy. Błąd bez postępu przerywa pracę.
Dziennik `execution/` przechowuje hash launchera, czasy, kody wyjścia i postęp.

Zmiana kodu obliczeniowego, parametrów lub środowiska, w tym wersji macOS,
wymaga nowej nazwy przebiegu. Nie należy ręcznie zmieniać manifestów w celu
ominięcia kontroli. Zmiana danych wymaga także zarchiwizowania poprzedniego
`data/v2/` przed ponownym przygotowaniem zbioru.

## Artefakty i wizualizacje

Wyniki są zapisywane w `results/runs/<run-id>/seed-<seed>/`.

| Artefakt | Zawartość |
|---|---|
| `saved_models/` | Modele i manifesty treningu |
| `threshold_selection/`, `ensemble_selection/` | Kalibratory, progi, konfiguracja ensemble i wyniki walidacji |
| `cross_validation/<model>/<protocol>/` | Przydziały foldów, predykcje OOF, metryki i przedziały ufności |
| `final_test_metrics.csv` | Metryki końcowe i przedziały ufności |
| `final_test_predictions.csv` | Predykcje powiązane z identyfikatorami wiadomości |
| `final_test_slice_metrics.csv` | Ocena w podgrupach |
| `final_test_reliability_*.svg` | Wykresy kalibracji |
| `probability_distributions.png` | Rozkłady prawdopodobieństw obu klas |
| `error_analysis.csv`, `mcnemar_tests.csv` | Analiza błędów i porównania statystyczne |
| `study_status.json` | Kompletność artefaktów i status ograniczeń badania |

Etap `report` kontroluje kompletność; nie tworzy automatycznie raportu PDF/HTML.
Notebooki służą do analizy wyników i SHAP oraz wymagają kernela Jupyter/IPython
korzystającego ze zgodnego środowiska. Historyczne ilustracje w `assets/` nie
stanowią wyników protokołu v2.

## Weryfikacja implementacji

```bash
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen black --check .
uv run --frozen isort --check-only .
uv run --frozen mypy .

# Próba całego pipeline'u na małym lokalnym modelu CPU:
uv run --frozen python scripts/integration_smoke.py

# Krótka próba sprzętowa; wymaga pobranych checkpointów:
HF_HUB_OFFLINE=1 uv run --frozen python scripts/hardware_smoke.py
```

Testy obejmują m.in. grupowanie i podziały danych, akumulację ważonej straty,
odtwarzalność inicjalizacji, kalibrację, statystykę, integralność artefaktów
oraz izolację procesów. Próba integracyjna sprawdza również wznowienie,
addytywność SHAP i archiwizację; jej wyniki nie są wynikami badawczymi.
Jednokrokowy pomiar GPU nie określa szczytowego zużycia pamięci w długim treningu.

Archiwum danych można utworzyć i zweryfikować następująco:

```bash
uv run --frozen python -m src.data.archive data/releases/synthetic-v2.tar.gz
uv run --frozen python -m src.data.archive data/releases/synthetic-v2.tar.gz --verify
```

## Ograniczenia badania

- Etykiety pochodzą od generatorów; niezależny przegląd ekspercki pozostaje do wykonania.
- Brakuje pełnej dokumentacji promptów i parametrów generowania. Nazwy źródeł są
  deklaracjami pochodzenia, a `gpt_version_unverified` oznacza nierozstrzygniętą wersję.
- Źródła mają nierówne liczebności. Małe próby, zwłaszcza z niewielką liczbą
  phishingów, dają niestabilne oszacowania metryk.
- Korpus był wcześniej analizowany. Ponowny podział nie tworzy niezależnego testu
  potwierdzającego, a dobór progów i ensemble na walidacji ogranicza interpretację
  wyników walidacyjnych jako oceny generalizacji.
- Source-holdout bada transfer między deklarowanymi źródłami syntetycznymi;
  nie zastępuje oceny na rzeczywistych wiadomościach.
- Jeden seed nie pozwala oszacować zmienności wynikającej z losowości treningu.
  Ustawienie seedów nie gwarantuje bitowej zgodności obliczeń między platformami.

## Organizacja repozytorium i licencja

| Katalog | Przeznaczenie |
|---|---|
| `src/data/` | Parsowanie, walidacja, grupowanie, podziały i archiwizacja |
| `src/models/` | Baseline, trening, CV i konfiguracja urządzenia |
| `src/evaluation/` | Inferencja, kalibracja, statystyka, raportowanie i SHAP |
| `src/utils/` | Manifesty, sumy kontrolne, statusy i logowanie |
| `docs/` | Karta zbioru, pochodzenie danych i podsumowanie wyników |
| `scripts/` | Uruchamianie badań i próby integracyjne/sprzętowe |
| `notebooks/` | Analizy interaktywne |
| `tests/` | Testy regresji |

Kod jest udostępniany na licencji [MIT](LICENSE). Licencja kodu nie określa
warunków udostępniania zbioru danych ani modeli bazowych.
