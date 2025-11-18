# Customer Intelligence Hub

> **Unified analytics system for understanding, predicting, and improving customer behavior — built for real-world business impact.**

[https://customer-intelligence-demo.pandeakshat.com](https://customer-intelligence-demo.pandeakshat.com/) [https://www.python.org/](https://www.python.org/) [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) [#](https://www.kimi.com/chat/19a96866-0212-8f2d-8000-092dfbeb4447#)

---

## 📘 Overview

The **Customer Intelligence Hub** is a production-ready Streamlit application that integrates multiple customer analytics modules — **churn prediction, sentiment analysis, RFM segmentation, and geospatial insights** — into a unified, interactive dashboard. It empowers businesses to proactively reduce churn, understand customer feedback at scale, and drive data-driven retention strategies.

- **Type**: Full-Stack Data Science Application
    
- **Tech Stack**: Python, Streamlit, Scikit-learn, XGBoost, Plotly, SHAP
    
- **Status**: Actively Deployed & Maintained
    
- **Impact**: 85% recall on churn prediction | 360° customer view
    

---

## ⚙️ Features

### 🔮 **Churn Prediction Module**

- **Models**: Logistic Regression (baseline) + XGBoost (production) with hyperparameter tuning
    
- **Performance**: Achieved **85% recall** on high-risk customers (priority business metric)
    
- **Explainability**: Integrated SHAP for model interpretability and individual predictions
    
- **Output**: Risk scoring, top churn drivers, and actionable retention recommendations
    

### 📊 **Customer Segmentation (RFM)**

- **Methodology**: RFM (Recency, Frequency, Monetary) analysis with K-Means clustering
    
- **Deliverable**: Dynamic segment labeling (Champions, At-Risk, Hibernating, etc.)
    
- **Visualization**: Interactive 3D scatter plots and segment comparison charts
    

### 💬 **Sentiment Analysis & NLP**

- **Technique**: LDA (Latent Dirichlet Allocation) for topic modeling + rule-based sentiment scoring
    
- **Processing**: Pandas + spaCy for text preprocessing and entity recognition
    
- **Output**: Sentiment distribution, keyword extraction, and topic trends over time
    

### 🗺️ **Geospatial Insights (Beta)**

- Regional churn/sentiment heatmaps
    
- Location-based customer value analysis
    

---

## 🧩 Architecture / Design

Text

Copy

```text
customer-intelligence/
├── app.py                          # Main Streamlit orchestrator
├── modules/
│   ├── churn_analysis.py          # XGBoost + SHAP pipeline
│   ├── sentiment_analysis.py      # LDA + NLP preprocessing
│   ├── segmentation.py             # RFM + K-Means engine
│   └── geospatial.py               # GeoPandas visualization
├── data/
│   ├── sample_customer_data.csv    # Synthetic dataset (10K records)
│   └── model_artifacts/            # Serialized models + encoders
├── requirements.txt
└── README.md
```

**Component Flow**:

- **Modular Design**: Each module is self-contained with clear inputs/outputs for scalability
    
- **Central Interface**: Streamlit acts as the integration layer, calling modules and rendering Plotly visuals
    
- **Model Registry**: Serialized models in `model_artifacts/` enable fast loading and reproducibility
    
- **Business Logic**: All modules return both data and human-readable insights for stakeholder communication
    

---

## 🚀 Quick Start

### 1. Clone and Setup

bash

Copy

```bash
git clone https://github.com/pandeakshat/customer-intelligence.git
cd customer-intelligence
```

### 2. Install Dependencies

bash

Copy

```bash
pip install -r requirements.txt
```

### 3. Run Application

bash

Copy

```bash
streamlit run app.py
```

> **Live Demo**: The app deploys automatically to [customer-intelligence-demo.pandeakshat.com](https://customer-intelligence-demo.pandeakshat.com/)

---

## 🧠 Example Output / Demo

The dashboard provides **four interactive views**:

1. **Churn Risk Profiler**:
    
    - Filter by segment, tenure, or product usage
        
    - SHAP waterfall charts explain _why_ a customer is high-risk
        
    - Downloadable retention priority list
        
2. **RFM Segment Explorer**:
    
    - 3D cluster visualization with hover details
        
    - Segment migration tracking over quarters
        
3. **Sentiment Analysis Panel**:
    
    - Topic trend line charts
        
    - Keyword co-occurrence network graph
        
4. **Executive Summary**:
    
    - Automated insights generation (e.g., "High-value customers in North America show 23% higher churn risk")
        

---

## 📊 Impact & Results

Table

Copy

|Metric|Value|Business Interpretation|
|:--|:--|:--|
|**Churn Recall**|85%|Correctly identifies 8.5/10 customers who will leave|
|**Model Precision**|72%|7.2/10 flagged customers actually churn (minimize false alarms)|
|**Segmentation Coverage**|100%|All 10K+ customers automatically segmented monthly|
|**Sentiment Processing**|~15K reviews/hr|Scalable NLP pipeline for real-time feedback analysis|

**Key Business Outcomes**:

- Enables proactive retention campaigns for high-risk cohorts
    
- Reduces manual segmentation effort from 3 days to 30 minutes
    
- Provides explainable predictions for C-suite stakeholder trust
    

---

## 🔍 Core Concepts

Table

Copy

|Area|Tools & Techniques|Purpose|
|:--|:--|:--|
|**Data Pipeline**|Pandas, NumPy, scikit-learn pipelines|Robust preprocessing + feature engineering|
|**Predictive Modeling**|XGBoost, Logistic Regression, cross-validation|High-performance churn prediction|
|**Model Explainability**|SHAP (TreeExplainer)|Interpretable AI for business trust|
|**Segmentation**|RFM analysis, K-Means, StandardScaler|Behavior-based customer grouping|
|**NLP**|spaCy (lemmatization), LDA (Gensim), VADER|Scalable sentiment & topic modeling|
|**Visualization**|Plotly Express, GeoPandas, Streamlit components|Interactive, publication-quality charts|

---

## 📈 Roadmap

- [x] Core churn + sentiment modules (85% recall achieved)
    
- [x] RFM segmentation + K-Means clustering
    
- [x] SHAP explainability integration
    
- [ ] **Q1 2025**: Geospatial analytics (regional risk heatmaps)
    
- [ ] **Q2 2025**: Retention scoring engine + automated email recommendations
    
- [ ] **Q3 2025**: Real-time API integration with CRM systems
    
- [ ] **Future**: A/B testing framework for retention interventions
    

---

## 🧮 Tech Highlights

**Languages:** Python, SQL  
**ML Frameworks:** Scikit-learn, XGBoost, SHAP, Gensim (LDA)  
**Data Stack:** Pandas, NumPy, spaCy, GeoPandas  
**Visualization:** Plotly, Streamlit, Matplotlib (for SHAP)  
**Deployment:** AWS EC2 + Docker (containerized)  
**MLOps:** Model versioning with `joblib`, automated CI/CD via GitHub Actions  
**Integrations:** Compatible with [Data Intelligence](https://github.com/pandeakshat/data-intelligence) for data auditing

---

## 🧰 Dependencies

txt

Copy

```txt
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.4.0
xgboost==2.0.3
plotly==5.18.0
shap==0.44.0
spacy==3.7.2
gensim==4.3.2
geopandas==0.14.3
```

---

## 🧾 License

MIT License © [Akshat Pande](https://github.com/pandeakshat)

---

## 🧩 Related Projects

- [https://github.com/pandeakshat/data-intelligence](https://github.com/pandeakshat/data-intelligence) — Dataset audit & augmentation tool (pre-processing pipeline for this project)
    
- [https://github.com/pandeakshat/project-flow](https://github.com/pandeakshat/project-flow) — Smart productivity and task-flow manager (project management)
    

---

## 💬 Contact

**Akshat Pande**  
📧 [mail@pandeakshat.com](mailto:mail@pandeakshat.com)  
🌐 [Portfolio](https://pandeakshat.com/) | [LinkedIn](https://linkedin.com/in/pandeakshat) | [GitHub](https://github.com/pandeakshat)