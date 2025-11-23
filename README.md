# Customer Intelligence Hub (v1.0)

> **Unified analytics system for understanding, predicting, and improving customer behavior — engineered with an API-Ready Monolith architecture.**

[![Demo App](https://img.shields.io/badge/Demo-Live_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://customer-intelligence-demo.pandeakshat.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📘 Overview

The **Customer Intelligence Hub** has evolved into a production-grade **Version 1** solution. It is no longer just a collection of scripts, but a cohesive **Full-Stack Data Science Application** designed with a strict separation of concerns (Logic vs. UI).

This system integrates four core engines—**Churn Simulation, Strategic Segmentation, Split-Topic Sentiment, and Contextual Geospatial Analysis**—into a single "Smart" dashboard that auto-detects dataset capabilities and self-heals dirty data.

---

## ⚙️ The "Smart" Core

Unlike standard dashboards, this application features intelligent middleware that abstracts complexity from the user:

* **The Modular Validator**: The "Gatekeeper" that scans uploaded files (CSV/JSON). It detects which columns are present and automatically enables or disables specific modules (e.g., *found 'Lat/Lon'? Enable Geospatial. Found 'Review'? Enable Sentiment*).
* **Smart Rename & Loader**: A universal data loader that standardizes disparate inputs (e.g., mapping user columns like "Amt" or "Bill" to system standard `TotalAmount`) and handles I/O.
* **Self-Healing Pipelines**: The logic layer automatically fills missing values, fixes data types, and handles empty strings before they break the model.

---

## 🚀 The 4 Intelligence Engines

### 1. 🔮 Churn Prediction & Simulation
* **Logic**: XGBoost pipeline with "Self-Healing" preprocessing.
* **The Simulator**: A "What-If" interface allowing stakeholders to tweak variables (e.g., *change Contract from Month-to-Month to One-Year*) and watch the Risk Score drop in real-time.
* **Directional Importance**: Visualization that shows not just *what* matters, but *how* it matters (e.g., Green bars = lowers risk, Red bars = increases risk).

### 2. 📊 Strategic Segmentation (RFM+)
* **Unified Engine**: Supports both Demographic clustering and RFM (Recency, Frequency, Monetary) analysis.
* **Rule Extraction**: Uses a Decision Tree overlaid on K-Means clusters to generate plain English rules (e.g., *"Cluster 1 is defined by Age < 30 & Spend > $500"*).
* **Recommendation Engine**: Translates mathematical Cluster IDs into human personas (e.g., "Gen Z Trendsetter") and suggests strategic actions.

### 3. 💬 Sentiment & Voice of Customer
* **Split-Topic Analysis**: Unlike standard LDA, this separates topics by sentiment. It identifies exactly what drives *Positive* reviews (e.g., "Fast Service") vs. *Negative* reviews (e.g., "Hidden Fees").
* **Correlation Matrix**: Statistically identifies which specific sub-rating (Food, Service, Ambiance) has the highest impact on the Overall Rating.

### 4. 🗺️ Geospatial Intelligence ("Piggyback")
* **Context-Aware**: This engine does not require its own dataset. It "piggybacks" onto Churn or Sentiment data.
    * *If Churn Data detected:* Plots a **Risk Heatmap**.
    * *If Sentiment Data detected:* Plots a **Happiness Map**.
* **Route Parsing**: Capable of parsing route/transportation data for logistics context.

---

## 🧩 Architecture / Design

We successfully transitioned from a script-based prototype to an **API-Ready Monolith**.

```text
customer-intelligence/
├── app.py                   # The Orchestrator (Auto-detects capabilities)
├── src/                     # PURE LOGIC (No UI Code)
│   ├── data_loader.py       # Handles I/O & Smart Renaming
│   ├── validator.py         # The Gatekeeper (Auto-detects columns)
│   ├── churn_engine.py      # XGBoost + SHAP + Simulator
│   ├── segment_engine.py    # K-Means + Rule Extraction
│   ├── sentiment_engine.py  # VADER + Split-Topic LDA
│   ├── geo_engine.py        # Piggyback Context Mapper
│   └── recommendation_engine.py # Business Logic & Personas
├── pages/                   # PURE UI (Streamlit widgets only)
│   ├── 1_Churn_Profiler.py
│   ├── 2_RFM_Segmentation.py
│   ├── 3_Sentiment_Analysis.py
│   └── 4_Geospatial_View.py
├── data/
│   └── model_artifacts/     # Serialized models
└── requirements.txt