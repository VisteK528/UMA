# Uczenie Maszynowe
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)

## Spis Treści
* [Opis](#opis)
* [Struktura projektu](#struktura-projektu)
* [Instalacja](#instalacja)
* [Eksperymenty](#eksperymenty)

## Opis
To repozytorium zawiera implementacje algorytmów uczenia maszynowego oraz eksperymenty wykonane w ramach przedmiotu Uczenie Maszynowe na Wydziale Elektroniki i Technik Informacyjnych Politechniki Warszawskiej. 
Zaimplementowane algorytmy:
- Drzewo decyzyjne ID3 (bez przycinania)
- Naiwny klasyfikator bayesowski (z dwoma typami dyskretyzacji)
- Zmodyfikowany Las Losowy (z możliwością wyboru procentu drzew, pozostała część uzupełniona przez NBC)

## Struktura projektu
- `data_processed` - przetworzone dane dla eksperymentów
- `experiments` - kod eksperymentów i ich wyniki
- `uma24z-nbc-random-forest` - pakiet Python zawierający implementacje algorytmów uczenia maszynowego

## Instalacja
Pakiet wymaga następujących bibliotek:
- `numpy` - wersja `1.2` lub nowsza
- `scikit-learn` - wersja `1.5.1` lub nowsza
- `matplotlib` - wersja `3.9.0` lub nowsza
- `joblib` - wersja `1.4.2` lub nowsza

Aby zainstalować pakiet, należy użyć następujących komend:
```bash
git clone https://github.com/VisteK528/UMA.git
cd UMA/
pip install uma24z-nbc-random-forest/
```

## Eksperymenty
Eksperymenty zostały przeprowadzone na 4 różnych zbiorach danych:
- `wine`
- `diabetes`
- `healthcare`
- `credit_score`

Zadanie klasyfikacji przeprowadzono na każdym ze wspomnianych zbiorów, używając 4 różne modele:
- `Naive Bayes Classifier`
- `ID3 decision tree` z głębokością dobieraną indywidualnie dla każdego zbioru danych
- `Classic Random Forest` z 50 modelami lokalnymi (same drzewa ID3)
- `Modified Random Forest` z 50 modelami lokalnymi (25 drzew ID3 i 25 NBC)