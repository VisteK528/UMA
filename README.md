# Machine Learning
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)

## Table of Contents
* [Overview](#overview)
* [Project structure](#project-structure)
* [Installation](#installation)
* [Unit tests](#unit-tests)
* [Experiments](#experiments)

## Overview
This repository contains implementation of machine learning algorithms and experiments done for the Machine Learning course at Faculty of Electronics and Information Technology, Warsaw University of Technology.
Implemented algorithms:
- ID3 decision tree (without pruning)
- Naive Bayes Classifier (with two types of discretization)
- Modified Random Forest (with possibility to select percentage of trees, the remaining part will be filled with NBCs)

## Project structure
- `data_processed` - preprocessed data for the experiments
- `experiments` - experiments code and their results
- `uma24z-nbc-random-forest` - python package containing machine learning implementations

## Installation
Package depends on following packages:
- `numpy` - version `1.2` and greater
- `scikit-learn` - version `1.5.1` and greater
- `matplotlib` - version `3.9.0` and greater
- `joblib` - version `1.4.2` and greater

To install the package run these commands from the parent directory
```bash
git clone https://github.com/VisteK528/UMA.git
cd UMA/
pip install uma24z-nbc-random-forest/
```

## Unit tests
After the installation process is completed one can run unit tests if want by running following command from the parent directory:

```bash
pytest uma24z-nbc-random-forest/tests/
```

## Experiments 
Experiments were conducted on 4 unique datasets:
- `wine`
- `diabetes`
- `healthcare`
- `credit_score`

Classification task was performed on each of them using 4 models:
- `Naive Bayes Classifier`
- `ID3 decision tree` with depth selected for each dataset individually
- `Classic Random Forest` with 50 local models (all ID3)
- `Modified Random Forest` with 50 local models (25 ID3s and 25 NBCs)

