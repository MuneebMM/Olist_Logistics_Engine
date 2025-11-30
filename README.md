# 🇧🇷 Olist Logistics Engine: AI-Powered Supply Chain Optimization

![Olist Logistics](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)

## 📖 Executive Summary

The **Olist Logistics Engine** is an end-to-end data science solution designed to predict and mitigate delivery delays in the Brazilian e-commerce landscape. By analyzing over 100k orders, we've built a robust machine learning pipeline that identifies high-risk shipments _before_ they happen, allowing for proactive intervention.

This project transforms raw data into actionable intelligence, featuring a production-ready **XGBoost** model, a **Streamlit** command center for logistics managers, and a scalable **MLOps** architecture.

---

## 🏗️ Project Architecture

We restructured a flat collection of notebooks into a modular, production-grade system:

```
Olist_Logistics_Engine/
├── 📂 api/                 # Future-proof API endpoints (Flask/FastAPI)
├── 📂 data/                # Data Lakehouse architecture
│   ├── 📂 raw/             # Immutable raw CSVs
│   └── 📂 processed/       # SQLite Master Analytics Table
├── 📂 models/              # Serialized model artifacts & encoders
├── 📂 notebooks/           # Experimental sandbox (EDA, Prototyping)
├── 📂 src/                 # Core Logic
│   ├── 📜 database.py      # ETL & Data Warehousing
│   ├── 📜 features.py      # Feature Engineering & Preprocessing
│   ├── 📜 train.py         # ML Training Pipeline (with MLflow)
│   └── 📜 inference.py     # Real-time Prediction Engine
├── 🐳 Dockerfile           # Containerization for deployment
├── 📜 streamlit_app.py     # Interactive Logistics Dashboard
└── 📜 requirements.txt     # Dependency management
```

---

## 🔍 Key Insights & Findings

Through rigorous Exploratory Data Analysis (EDA), we uncovered critical bottlenecks in the supply chain:

1.  **The "Last Mile" is Long**: Distance is a primary driver of delays, but not linear. Cross-state shipments (e.g., SP to North) have exponentially higher failure rates due to infrastructure challenges.
2.  **Seller Performance Variance**: A small cluster of sellers contributes disproportionately to late deliveries. "Seller Late Rate" proved to be the most predictive feature in our model.
3.  **The "Handover" Gap**: Significant time is often lost between `order_approved` and `carrier_pickup`. Optimizing this window offers the highest ROI for reducing total delivery time.
4.  **Customer Satisfaction Correlation**: There is a sharp cliff in review scores once an order is late. On-time delivery is the single biggest predictor of 5-star reviews.

---

## 🛠️ The Solution: Predictive Logistics

We implemented a **Gradient Boosted Decision Tree (XGBoost)** classifier to predict the probability of an order being late (`is_late = 1`).

### Feature Engineering

- **Geospatial**: Calculated Haversine distance between Seller and Customer.
- **Seller Profiling**: Historical "Late Rate" computed on a rolling basis (preventing data leakage).
- **Product Attributes**: Weight and Freight cost as proxies for logistics complexity.

### Performance

- **ROC-AUC**: ~0.73 (Good discrimination between on-time and late orders).
- **Recall**: Optimized to catch ~60-70% of potential delays, prioritizing "Safety" over precision to ensure high-risk orders are flagged.

---

## 🚀 Strategic Recommendations

Based on our model's predictions, we recommend the following actions for Olist:

1.  **Proactive Customer Communication**: For orders flagged as **High Risk (>50%)**, automatically notify the customer of a potential delay _at purchase time_ or offer a discount for patience. This manages expectations and protects NPS.
2.  **Seller Intervention Program**: Target the bottom 10% of sellers (identified by our `Seller Late Rate` feature) for operational audits. If they fail to improve handover times, deprioritize their listings.
3.  **Dynamic Carrier Routing**: For long-distance, high-weight shipments (high risk factors), route to premium carriers automatically, even at a slightly higher cost, to avoid the massive CSAT hit of a late delivery.
4.  **Inventory Distribution**: Incentivize top-selling products to be stored in fulfillment centers closer to high-demand regions (e.g., placing stock in the Northeast) to reduce the `distance_km` factor.

---

## 💻 How to Run the Engine

### 1. Setup Environment

```bash
# Clone the repo
git clone <repo-url>
cd Olist_Logistics_Engine

# Install dependencies
pip install -r requirements.txt
```

### 2. Build the Data Warehouse

Load raw CSVs into SQLite and create the Master Analytics Table.

```bash
python -c "from src.database import load_raw_data, create_master_table; load_raw_data(); create_master_table()"
```

### 3. Train the Model (MLOps)

Train the XGBoost model and log experiments to MLflow.

```bash
python -m src.train
# View experiments
mlflow ui
```

### 4. Launch the Dashboard

Start the Logistics Command Center.

```bash
streamlit run streamlit_app.py
```

### 5. Docker Deployment (Optional)

```bash
docker build -t olist-dashboard .
docker run -p 8501:8501 olist-dashboard
```

---

## 🔮 Future Roadmap

- **Deep Learning**: Implement LSTM/GRU for time-series forecasting of daily order volume.
- **Real-time Ingestion**: Move from SQLite to Snowflake/BigQuery with Airflow orchestration.
- **A/B Testing**: Test the "Proactive Notification" strategy on a subset of users to measure impact on cancellation rates.

---

_Built with ❤️ by the Olist Data Science Team_
