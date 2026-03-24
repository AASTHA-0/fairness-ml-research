# Fairness in Machine Learning – COMPAS Dataset Analysis

This project analyzes potential bias in the COMPAS recidivism prediction dataset.

## Dataset
COMPAS Scores dataset

# FairSight 🔍⚖️
### Detecting and Mitigating Algorithmic Bias in Criminal Justice Recidivism Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)
![Fairness](https://img.shields.io/badge/Focus-Algorithmic%20Fairness-red)
![Dataset](https://img.shields.io/badge/Dataset-COMPAS-purple)

---

## 🚨 The Problem

> *"An algorithm decides if you go to jail or go home.*
> *And that algorithm is biased."*

- 🛒 **Amazon, 2018** — AI rejected women's CVs for 4 years
- 💳 **Apple Card, 2019** — Wives got 20x less credit than husbands
- ⚖️ **COMPAS, USA** — Black defendants labeled high-risk at 2x the rate

**Bias is not theoretical. It is destroying lives — silently.**

FairSight is a complete framework to **detect, measure, and mitigate** algorithmic bias in ML models.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Bias Detection Methods](#bias-detection-methods)
- [Bias Mitigation](#bias-mitigation)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [References](#references)

---

## 🧠 Overview

FairSight is a research framework that:

- ✅ Detects racial and gender bias in ML predictions
- ✅ Identifies proxy variables that carry hidden bias
- ✅ Mitigates bias through targeted feature elimination
- ✅ Measures fairness using 4 industry-standard metrics
- ✅ Visualizes before/after bias comparison clearly

> **Research presented at ABHIGYA-2026**
> National Institute of Technology Hamirpur, H.P., India

---

## 📊 Dataset

**COMPAS Recidivism Dataset**
Source: [ProPublica GitHub](https://github.com/propublica/compas-analysis)

| Property | Value |
|---|---|
| Total Records | 7,214 defendants |
| Location | Broward County, USA |
| Target Variable | two_year_recid (binary) |
| Class 0 — No Recidivism | 4,357 (60.4%) |
| Class 1 — Recidivism | 2,857 (39.6%) |
| Train / Test Split | 70% / 30% |

**Features Used:**
age, priors_count, juv_fel_count,
juv_misd_count, c_charge_degree,
days_b_screening_arrest

race  # 6 categories: African-American, Caucasian,
      # Hispanic, Asian, Native American, Other
sex   # binary: Male=1, Female=0
```

---

## 🔍 Bias Detection Methods

### 1️⃣ Demographic Parity
```
P(Ŷ=1 | A=a) ≠ P(Ŷ=1 | A=b)
```
If prediction rates differ across groups → bias exists.

---

### 2️⃣ Disparate Impact (DI)
```
DI = P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)
```

| DI Value | Status |
|---|---|
| < 0.8 | ❌ Biased against unprivileged |
| > 1.25 | ❌ Biased against privileged |
| 0.8 – 1.25 | ✅ Fair |

---

### 3️⃣ False Positive Rate (FPR)
```
FPR = FP / (FP + TN)
```
Measures how often innocent people are
wrongly labeled as high-risk.

---

### 4️⃣ Mutual Information (MI)
```
I(S;Y) = Σ P(s,y) log [P(s,y) / P(s)·P(y)]

#proxy bias----->
# Correlation Analysis — Hidden Bias Channels
race vs priors_count = -0.191  # significant
sex  vs priors_count = +0.120  # significant

#Final Fair Feature Set:
features = [
    'age',
    'juv_fel_count',
    'c_charge_degree',
    'days_b_screening_arrest'
]


**requirements.txt:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
kaggle



