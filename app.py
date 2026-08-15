import streamlit as st
import joblib
import pandas as pd
from utils import build_feature_row, haversine_distance
from pathlib import Path

st.set_page_config(page_title="Zomato ETA Predictor", page_icon="🛵", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #FFF5F3 0%, #FFFFFF 35%);
    font-family: 'Segoe UI', sans-serif;
}
.hero-banner {
    background: linear-gradient(135deg, #CB202D 0%, #E23744 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(203,32,45,0.25);
}
.hero-banner h1 { color: white !important; margin-bottom: 0.2rem; }
.hero-banner p { color: #FFE5E5; font-size: 1.05rem; margin: 0; }
.model-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.3em 0.9em;
    border-radius: 20px;
    margin-top: 0.6rem;
}
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-left: 5px solid #CB202D;
}
h3 { color: #CB202D; font-weight: 700; }
div.stButton > button {
    background: linear-gradient(135deg, #CB202D 0%, #E23744 100%);
    color: white; font-weight: 700; font-size: 1.05rem;
    border-radius: 10px; padding: 0.7em 1.5em; border: none;
    box-shadow: 0 4px 12px rgba(203,32,45,0.35);
}
div.stButton > button:hover { background: #A5181F; }
[data-testid="stMetricValue"] { color: #CB202D; font-weight: 700; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('zomato_delivery_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, feature_cols

model, feature_cols = load_model()

if Path("zomato_logo.png").exists():
    logo_col, _ = st.columns([1, 6])
    with logo_col:
        st.image("zomato_logo.png", width=70)

st.markdown("""
<div class="hero-banner">
    <h1>🛵 Delivery Time Predictor</h1>
    <p>Predicts ETA using traffic, weather, distance, and rider context — not distance alone.</p>
    <span class="model-badge">⚡ Powered by XGBoost · Test MAE 3.04 min · R² 0.837</span>
</div>
""", unsafe_allow_html=True)

st.caption("📊 See the **Dashboard** page in the left sidebar for full model comparison, EDA, and business insights.")

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🧾 Order Details")
oc1, oc2, oc3 = st.columns(3)
with oc1:
    order_type = st.selectbox("Order Type", ['Snack', 'Meal', 'Drinks', 'Buffet'])
with oc2:
    festival = st.selectbox("Festival Period?", ['No', 'Yes'])
with oc3:
    order_hour = st.slider("Order Hour (24h)", 0, 23, 19)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🌦️ Conditions")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
with cc2:
    weather = st.selectbox("Weather", ['Sunny', 'Stormy', 'Windy', 'Fog', 'Sandstorms', 'Cloudy'])
with cc3:
    city = st.selectbox("City Type", ['Urban', 'Semi-Urban', 'Metropolitan'])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🏍️ Rider & Vehicle")
rc1, rc2, rc3 = st.columns(3)
with rc1:
    age = st.slider("Rider Age", 18, 60, 30)
    vehicle_type = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'bicycle', 'electric_scooter'])
with rc2:
    ratings = st.slider("Rider Rating", 1.0, 5.0, 4.5)
    vehicle_condition = st.selectbox("Vehicle Condition (0=poor, 3=excellent)", [0, 1, 2, 3])
with rc3:
    multiple_deliveries = st.selectbox("Simultaneous Deliveries", [0, 1, 2, 3])
    st.caption("⚠️ 3+ deliveries at once tends to increase ETA")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📍 Locations")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Restaurant**")
    rest_lat = st.number_input("Latitude", value=12.97, key="rl")
    rest_lon = st.number_input("Longitude", value=77.59, key="rlo")
with c2:
    st.markdown("**Delivery Address**")
    deliv_lat = st.number_input("Latitude", value=12.93, key="dl")
    deliv_lon = st.number_input("Longitude", value=77.61, key="dlo")

live_distance = haversine_distance(rest_lat, rest_lon, deliv_lat, deliv_lon)
st.info(f"📏 Calculated distance: **{live_distance:.2f} km**")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔮 Predict Delivery Time", type="primary", width='stretch'):
    inputs = {
        'age': age, 'ratings': ratings, 'vehicle_condition': vehicle_condition,
        'multiple_deliveries': multiple_deliveries, 'festival': festival,
        'traffic': traffic, 'weather': weather, 'order_type': order_type,
        'vehicle_type': vehicle_type, 'city': city, 'order_hour': order_hour,
        'rest_lat': rest_lat, 'rest_lon': rest_lon,
        'deliv_lat': deliv_lat, 'deliv_lon': deliv_lon,
    }
    row = build_feature_row(inputs, feature_cols)
    prediction = model.predict(row)[0]

    st.success(f"### 🕒 Estimated Delivery Time: {prediction:.1f} minutes")

    notes = []
    if traffic in ['High', 'Jam']:
        notes.append(f"🚦 **{traffic} traffic** is pushing this estimate up — traffic is the single strongest ETA driver in this model.")
    if weather in ['Stormy', 'Fog', 'Sandstorms']:
        notes.append(f"🌧️ **{weather} weather** adds delay compared to clear conditions.")
    if multiple_deliveries >= 3:
        notes.append("📦 This rider is carrying **3+ simultaneous deliveries** — a known factor that increases delivery time.")
    if live_distance > 15:
        notes.append(f"📏 This is a **long-distance delivery** ({live_distance:.1f} km).")
    if not notes:
        notes.append("✅ Conditions are favorable — low traffic, clear weather, single delivery.")
    for n in notes:
        st.markdown(f"- {n}")
