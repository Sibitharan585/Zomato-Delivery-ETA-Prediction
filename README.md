# Beyond Distance: Predicting Food Delivery Time with Machine Learning

Food delivery platforms like Zomato and Swiggy estimate delivery time primarily using
distance, but distance alone explains only 32% of the variation in actual delivery time
(r = 0.32). Real-world factors — traffic density, weather conditions, rider workload, and
time of day — matter more, yet static distance-based formulas ignore them. Inaccurate
ETAs lead to cold food, order cancellations, poor customer reviews, and reduced customer
lifetime value.

This project builds a machine learning model that predicts delivery time using order,
rider, traffic, and weather context — not distance alone — and proves this claim
empirically using correlation analysis, model benchmarking, and SHAP explainability.

## Key Results

| Metric | Value |
|---|---|
| Final model | Tuned XGBoost Regressor |
| Test MAE | 3.04 minutes (target was under 5 minutes) |
| R² Score | 0.837 |
| Validation | 10-fold CV, train/test overfitting check, chronological future-orders test |

## What Makes This Different

Rather than assuming distance drives delivery time, this project proves it doesn't:
distance alone correlates at only **r = 0.32** with delivery time. SHAP explainability
confirms traffic density, rider workload, and rider rating matter more — directly
justifying why a smarter model beats a distance-only ETA formula.

## Data Pipeline

- Fixed 3 real data-quality bugs found during EDA: GPS sign-flip errors on restaurant
  and delivery coordinates, an Excel time-format bug affecting ~9% of order timestamps,
  and rating-scale outliers (values of 6 on a 1–5 scale)
- Engineered features: Haversine great-circle distance, rush-hour flag, order-hour
  extraction, multiple-deliveries load
- Ordinal encoding for ordered categories (traffic density), one-hot encoding for
  nominal categories (weather, vehicle type, city, order type)
- 80/20 train-test split, 5 baseline models benchmarked, XGBoost tuned via
  RandomizedSearchCV

## Model Validation

- **Overfitting check:** train MAE 2.83 min vs test MAE 3.04 min — a gap of only 0.22
  minutes, confirming the model generalizes rather than memorizes
- **10-fold cross-validation:** stable across all folds (std ≈ 0.03 min)
- **Chronological holdout:** trained on earlier orders, tested on later/unseen orders —
  performance held within 0.1 minutes of the random-split result
- **SHAP explainability:** confirms traffic density, rider age, distance, and rider
  rating as the top predictive drivers

## App Structure

The Streamlit app has two pages:
- **Predictor (`app.py`)** — enter order, rider, traffic, weather, and location details
  to get a live delivery time prediction
- **Dashboard (`pages/1_Dashboard.py`)** — full EDA, model benchmarking comparison,
  and SHAP explainability, all interactive

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit · Plotly ·
Matplotlib · Seaborn

## Project Structure

```
├── Zomato_Delivery_Prediction.ipynb   # Full EDA & model pipeline (Colab)
├── app.py                              # Streamlit predictor (main page)
├── utils.py                            # Shared feature engineering logic
├── pages/
│   └── 1_Dashboard.py                  # EDA, model comparison & SHAP dashboard
├── zomato_delivery_model.pkl           # Trained model artifact
├── feature_columns.pkl                 # Exact feature order for inference
├── dashboard_data.csv                  # Sampled data for dashboard charts
├── model_comparison.csv                # Leaderboard of all benchmarked models
├── shap_importance.csv                 # SHAP feature importance values
├── architecture_diagram.png            # System architecture diagram
├── zomato_logo.png                     # App branding asset
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Built as part of the upGradX Capstone Project — Showcase Edition 2026.
