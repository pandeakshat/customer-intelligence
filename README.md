# Customer Intelligence Hub

**Customer Intelligence Hub** is a modular, end-to-end data science application built with Streamlit. It transforms raw customer data into actionable business strategies by integrating predictive modeling (XGBoost), unsupervised learning (K-Means), Natural Language Processing (VADER/LDA), and geospatial analysis into a unified interface.

The application features a "Piggyback" architecture, allowing independent analytic modules to activate automatically based on the detected schema of uploaded datasets.

## Core Capabilities

* **Churn Prediction Engine:** Uses XGBoost for classification and SHAP values for explainability. Features include a "Self-Healing" data cleaner and a real-time "What-If" simulator for testing retention strategies.
* **Segmentation Engine:** Implements K-Means clustering with an overlaid Decision Tree for rule extraction. Automatically detects analysis modes (Demographic vs. RFM) and generates descriptive "Smart Labels" for clusters.
* **Sentiment Analysis:** A hybrid NLP pipeline utilizing VADER for polarity scoring and Latent Dirichlet Allocation (LDA) for topic modeling to extract key themes from customer reviews.
* **Geospatial Intelligence:** A hybrid location engine that combines a static local database, fuzzy matching for spelling correction, and API-based geocoding to map customer performance globally.
* **Natural Language Generation (NLG):** A rule-based text generation system that converts statistical summaries into human-readable executive summaries.

## Technical Architecture

The project follows a component-based architecture:

* `app.py`: Central controller handling state management and module routing.
* `src/engines`: Isolated logic for Churn, Segmentation, Geo, and Sentiment.
* `src/components`: Reusable UI elements (Navigation, Data Loader).
* **State Management:** Robust usage of Streamlit Session State for data persistence across pages.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/customer-intelligence-hub.git](https://github.com/yourusername/customer-intelligence-hub.git)
    cd customer-intelligence-hub
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the application:
    ```bash
    streamlit run app.py
    ```

2.  **Upload Data:** Navigate to the Home page. You can upload your own CSV/Excel files or load the provided sample datasets for testing.
3.  **Validation:** The system will validate your schema. Green checkmarks indicate the module is ready.
4.  **Navigation:** Use the sidebar to access specific analytical engines (Churn, Segmentation, etc.).

## Dependencies

* Streamlit
* Pandas / NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Plotly
* Geopy

## License

Distributed under the MIT License. See LICENSE for more information.