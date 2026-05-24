<div align="center">

# 🧠 Synapse Drive

### ADAS Fuzzy Intelligence System

**A Hybrid Mamdani Fuzzy Logic + LSTM Neural Network for Real-Time Collision Risk Assessment**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Fuzzy](https://img.shields.io/badge/Scikit--Fuzzy-Mamdani_FIS-blue?style=for-the-badge)](https://pythonhosted.org/scikit-fuzzy/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Models Used](#-models-used)
- [Interactive Dashboard](#-interactive-dashboard)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

**Synapse Drive** is an Advanced Driver Assistance System (ADAS) that combines **soft computing techniques** — specifically **Mamdani Fuzzy Inference** and **LSTM Neural Networks** — to produce real-time collision risk assessments.

The system takes two sensor inputs:
- **Distance** to the lead vehicle (0–100 m)
- **Relative Speed** (0–100 km/h)

And produces three critical safety outputs:

| Output | Description |
|--------|-------------|
| 🛑 **Brake Force (%)** | How hard the brakes should be applied |
| ⚠️ **Collision Risk (%)** | Probability of a collision |
| 🔔 **Driver Alert (%)** | Warning level for the driver |

A **Hybrid Engine** fuses both models with adaptive weighting and a TTC safety override to produce the final risk assessment.

---

## ✨ Key Features

- 🧮 **Mamdani Fuzzy Inference System** — 9 expert IF-THEN rules with trapezoidal/triangular membership functions
- 🧠 **LSTM Neural Network** — 2-layer architecture (30,497 parameters) trained on temporal driving sequences
- ⚡ **Adaptive Hybrid Fusion** — Distance-dependent weighting (70/30 near, 60/40 far)
- 🛡️ **TTC Safety Override** — Hard safety floor when Time-To-Collision < 1.5 seconds
- 🔧 **Physics-Based Fallback** — Guarantees correct outputs at extreme edge cases
- 🖥️ **Interactive Web Dashboard** — Real-time visualization built with React 18 & Chart.js
- 📊 **Comprehensive Visualizations** — 3D inference surfaces, membership function plots, radar charts, prediction history

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────┐
                    │      SENSOR INPUTS       │
                    │  Distance (0-100m)       │
                    │  Rel. Speed (0-100 km/h) │
                    └─────────┬────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
    ┌────────────▼────────────┐  ┌─────────▼──────────┐
    │   MAMDANI FUZZY (FIS)   │  │   LSTM NETWORK     │
    │                         │  │                    │
    │ • 9 IF-THEN Rules       │  │ • 2 LSTM Layers    │
    │ • Trapezoidal/Tri MFs   │  │ • 10 Timesteps     │
    │ • Centroid Defuzz       │  │ • 3 Features       │
    │ • 3 Outputs (B/R/A)     │  │ • Sigmoid Output   │
    │                         │  │                    │
    │  ┌─── Physics Fallback  │  │  30,497 params     │
    └────────────┬────────────┘  └─────────┬──────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │  ADAPTIVE HYBRID ENGINE  │
                 │                          │
                 │  Near (<20m): 70F / 30L  │
                 │  Far (≥20m):  60F / 40L  │
                 │                          │
                 │  TTC Override: <1.5s →   │
                 │    force risk ≥ 75%      │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │      RISK CATEGORY       │
                 │  HIGH ≥ 70% │ MOD ≥ 40%  │
                 │       LOW < 40%          │
                 └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Fuzzy Logic Engine** | scikit-fuzzy (Mamdani FIS) |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Dashboard** | React 18, Chart.js, Vanilla CSS |
| **Notebook** | Jupyter / Google Colab |

---

## 🤖 Models Used

### 1. Mamdani Fuzzy Inference System (FIS)

A rule-based expert system using fuzzy logic with **degrees of membership** (0 to 1):

| Rule | IF Distance | IF Speed | THEN Brake | THEN Risk | THEN Alert | Interpretation |
|------|-------------|----------|------------|-----------|------------|----------------|
| R01 | Near | High | HIGH | HIGH | HIGH | Emergency brake |
| R02 | Near | Medium | HIGH | HIGH | MEDIUM | Hard brake |
| R03 | Near | Low | MEDIUM | MEDIUM | MEDIUM | Cautious |
| R04 | Medium | High | MEDIUM | HIGH | HIGH | Alert + moderate brake |
| R05 | Medium | Medium | MEDIUM | MEDIUM | MEDIUM | Normal traffic |
| R06 | Medium | Low | LOW | MEDIUM | LOW | Easy driving |
| R07 | Far | High | MEDIUM | MEDIUM | HIGH | Speed warning |
| R08 | Far | Medium | LOW | LOW | LOW | Safe |
| R09 | Far | Low | LOW | LOW | LOW | Very safe |

### 2. LSTM Neural Network (30,497 parameters)

| Layer | Units | Purpose |
|-------|-------|---------|
| LSTM (return_sequences=True) | 64 | Captures short-term temporal patterns |
| Dropout | 25% | Prevents overfitting |
| LSTM (return_sequences=False) | 32 | Compresses temporal information |
| BatchNormalization | — | Stabilizes training |
| Dropout | 20% | Additional regularization |
| Dense (ReLU) | 16 | Non-linear feature extraction |
| Dense (Sigmoid) | 1 | Collision probability [0, 1] |

### 3. Adaptive Hybrid Fusion Engine

```python
hybrid_risk = w_fuzzy × fuzzy_risk + w_lstm × (lstm_probability × 100)

# Adaptive weights based on distance
w_fuzzy = 0.70 if distance < 20m else 0.60
w_lstm  = 1 - w_fuzzy

# TTC Safety Override
if TTC < 1.5 seconds:
    hybrid_risk = max(hybrid_risk, 75.0)  # Force HIGH status
```

---

## 🖥️ Interactive Dashboard

The **Synapse Drive Dashboard** is a 3-page interactive web application:

| Page | Description |
|------|-------------|
| **Dashboard** | Real-time sensor input, fuzzy outputs, membership function visualization, rule firing table, radar chart, prediction history |
| **Analysis** | Scenario comparison charts, data tables, 3D inference surface heatmaps |
| **About** | System architecture documentation, complete Mamdani rule base |

### Running the Dashboard

Simply open the HTML file in any modern browser:

```bash
open adas_v3_blue.html
```

> No build tools or server required — the dashboard is a single self-contained HTML file with React 18 loaded via CDN.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-fuzzy tensorflow scikit-learn matplotlib seaborn
```

### Running the Jupyter Notebook

1. **Clone the repository:**
   ```bash
    git clone https://github.com/Arnavs10/Synapse-Drive.git
   cd Synapse-Drive
   ```

2. **Open the notebook:**
   ```bash
    jupyter notebook Synapse_Drive_ML_Pipeline.ipynb
    ```
   *Or upload to [Google Colab](https://colab.research.google.com) for cloud execution.*

3. **Run all cells** — the notebook trains the LSTM, runs fuzzy inference, and generates all visualizations.

### Running the Web Dashboard

```bash
open adas_v3_blue.html    # macOS
# or just double-click the file in your file manager
```

---

## 📁 Project Structure

```
Synapse-Drive/
│
├── Synapse_Drive_ML_Pipeline.ipynb
│   └── Complete Jupyter notebook — Fuzzy Logic + LSTM + Hybrid Engine
│
├── adas_v3_blue.html
│   └── Interactive Web Dashboard (React 18 + Chart.js, self-contained)
│
├── Screenshot 2026-04-20 at 4.55.07 PM.png
│   └── Dashboard screenshot
│
├── .gitignore
│
└── README.md
```

---

## 📊 Results

### LSTM Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~99.3% |
| **AUC Score** | ~1.0 |
| **Parameters** | 30,497 |
| **Training Sequences** | 3,000 |

> **Note:** The high accuracy is expected on synthetic data with well-separated clusters. This is a proof-of-concept — real-world ADAS data would yield 80–90% accuracy with noisy sensor inputs.

### Scenario Validation

| Scenario | Distance | Speed | Brake | Risk | Hybrid Category |
|----------|----------|-------|-------|------|-----------------|
| 🚨 Emergency | 5m | 90 km/h | 83.7% | 85.7% | **HIGH** |
| ⚠️ Dangerous | 20m | 70 km/h | 66.3% | 84.1% | **HIGH** |
| 🟡 Normal | 50m | 50 km/h | 48.3% | 51.7% | **MODERATE** |
| 🛣️ Highway | 80m | 80 km/h | 48.3% | 51.7% | **MODERATE** |
| 😌 Calm | 50m | 20 km/h | 11.7% | 51.7% | **LOW** |
| ✅ Safe | 95m | 10 km/h | 11.7% | 15.2% | **LOW** |

---

## 🙏 Acknowledgements

- [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) — Mamdani Fuzzy Inference System implementation
- [TensorFlow / Keras](https://tensorflow.org) — LSTM neural network framework
- [React](https://react.dev) — Interactive dashboard UI
- [Chart.js](https://www.chartjs.org/) — Dashboard charting library

---

<div align="center">

**Built with ❤️ using Soft Computing Techniques**

*Mamdani Fuzzy Logic × LSTM Neural Networks × Adaptive Hybrid Fusion*

</div>

