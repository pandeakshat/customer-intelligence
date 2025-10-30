# Customer Intelligence Hub

> Unified analytics system for understanding, predicting, and improving customer behavior.

---

## 📘 Overview

The Customer Intelligence Hub integrates multiple areas of customer analytics — churn prediction, sentiment analysis, segmentation, and geospatial insights — into one unified dashboard. It empowers businesses to understand customer behavior, improve retention, and make data-driven decisions. The app is modular, scalable, and built for applied analytics in real-world scenarios.

- Type: Streamlit App  
- Tech Stack: Python, Streamlit, Scikit-learn, Plotly  
- Status: Active  

---

## ⚙️ Features

- Unified dashboard for customer intelligence workflows.  
- Predictive modeling for churn and segmentation.  
- Sentiment analysis with NLP support.  

---

## 🧩 Architecture / Design

```text
customer-intelligence/
├── app.py
├── modules/
│   ├── churn_analysis.py
│   ├── sentiment_analysis.py
│   ├── segmentation.py
├── data/
│   └── sample.csv
└── README.md
```

Explain briefly how your components fit together:
- Each module handles one customer insight type (churn, sentiment, etc.).  
- Streamlit serves as the central interface integrating models and visuals.  
- Outputs include predictive charts, segment insights, and trend reports.

---

## 🚀 Quick Start

### 1. Clone and setup environment
```bash
git clone https://github.com/pandeakshat/customer-intelligence.git
cd customer-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

> The app will open locally at http://localhost:8501

---

## 🧠 Example Output / Demo

Displays an interactive dashboard for customer churn prediction, sentiment breakdowns, and segment-level insights.

> Example: “Highlights customer churn risk and sentiment distribution across categories.”

---

## 🔍 Core Concepts

| Area | Tools | Purpose |
|------|--------|----------|
| Data | Pandas, NumPy | Cleaning + preprocessing |
| Modeling | Scikit-learn | Churn and segmentation modeling |
| Visualization | Plotly, Streamlit | Interactive analysis |

---

## 📈 Roadmap

- [x] Core churn + sentiment modules  
- [ ] Add geospatial analytics  
- [ ] Extend to retention scoring and recommendations  
- [ ] Integrate with Data Intelligence Tool  

---

## 🧮 Tech Highlights

**Languages:** Python, SQL  
**Frameworks:** Streamlit, Scikit-learn, Plotly  
**Cloud:** AWS (optional)  
**Integrations:** Data Intelligence, ProjectFlow  

---

## 🧰 Dependencies

- streamlit  
- pandas  
- numpy  
- scikit-learn  
- plotly  

---

## 🧾 License

MIT License © [Akshat Pande](https://github.com/pandeakshat)

---

## 🧩 Related Projects

- [Data Intelligence](https://github.com/pandeakshat/data-intelligence) — Dataset audit & augmentation tool.  
- [Project Flow](https://github.com/pandeakshat/project-flow) — Smart productivity and task-flow manager.

---

## 💬 Contact

**Akshat Pande**  
📧 [mail@pandeakshat.com](mailto:mail@pandeakshat.com)  
🌐 [Portfolio](https://pandeakshat.com) | [LinkedIn](https://linkedin.com/in/pandeakshat)
