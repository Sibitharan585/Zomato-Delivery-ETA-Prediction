import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zomato ETA - Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #FFF8F5 0%, #FFFFFF 100%); }
h1, h2, h3 { color: #CB202D; }
[data-testid="stMetricValue"] { color: #CB202D; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Project Dashboard")
st.caption("Zomato Delivery Dataset — EDA, Model Comparison & SHAP Explainability")

dash_df = pd.read_csv('dashboard_data.csv')
shap_df = pd.read_csv('shap_importance.csv')
comparison_df = pd.read_csv('model_comparison.csv').sort_values('MAE (minutes)').reset_index(drop=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Model", "XGBoost")
m2.metric("Test MAE", "3.04 min", help="Average prediction error in minutes")
m3.metric("Test R²", "0.837", help="Fraction of delivery-time variation explained")
m4.metric("Orders Analyzed", f"{len(dash_df):,}")
st.divider()

st.subheader("🏆 Which model performed best?")
st.caption("6 algorithms were benchmarked on identical train/test splits. Lower MAE = more accurate.")
fig1 = px.bar(
    comparison_df, x='Model', y='MAE (minutes)', color='MAE (minutes)',
    color_continuous_scale='Reds_r', text='MAE (minutes)',
    title="Model Accuracy Comparison (lower is better)"
)
fig1.update_traces(texttemplate='%{text:.2f} min', textposition='outside')
fig1.update_layout(showlegend=False, yaxis_title="Mean Absolute Error (minutes)")
st.plotly_chart(fig1, width='stretch')
st.dataframe(comparison_df, width='stretch')

st.markdown("""
##### 🥇 Why XGBoost?

XGBoost outperformed all 5 other models tested — Random Forest, Decision Tree,
Gradient Boosting, Ridge, and Linear Regression — achieving the **lowest Test MAE
(3.04 minutes)** and **highest R² (0.837)**, and stayed stable across 10-fold
cross-validation with no meaningful overfitting (train/test MAE gap was small).

Delivery time depends on **non-linear interactions** between traffic, weather,
and rider context — relationships that linear models (Ridge, Linear Regression)
structurally can't capture. Tree-based ensemble methods like XGBoost model these
interactions naturally, which is exactly what the numbers above reflect.
""")
st.divider()

st.subheader("⏱️ How long do deliveries actually take?")
fig2 = px.histogram(
    dash_df, x='Time_taken (min)', nbins=30,
    title="Distribution of Delivery Times", color_discrete_sequence=['#CB202D']
)
fig2.add_vline(x=dash_df['Time_taken (min)'].mean(), line_dash="dash", line_color="black",
                annotation_text=f"Mean: {dash_df['Time_taken (min)'].mean():.1f} min")
fig2.update_layout(xaxis_title="Delivery Time (minutes)", yaxis_title="Number of Orders")
st.plotly_chart(fig2, width='stretch')
st.divider()

st.subheader("🚦 Does traffic actually slow deliveries down?")
traffic_labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Jam'}
dash_df['Traffic_Label'] = dash_df['Road_traffic_density'].map(traffic_labels)
fig3 = px.box(
    dash_df, x='Traffic_Label', y='Time_taken (min)',
    category_orders={'Traffic_Label': ['Low', 'Medium', 'High', 'Jam']},
    color='Traffic_Label', color_discrete_sequence=px.colors.sequential.Reds,
    title="Delivery Time by Traffic Density"
)
fig3.update_layout(xaxis_title="Traffic Density", yaxis_title="Delivery Time (minutes)", showlegend=False)
st.plotly_chart(fig3, width='stretch')
st.caption("📌 Clear upward trend: Jam traffic delivers noticeably slower than Low traffic, on average.")
st.divider()

st.subheader("📏 Is distance really the best predictor of delivery time?")
corr = dash_df['Distance_km'].corr(dash_df['Time_taken (min)'])
fig4 = px.scatter(
    dash_df, x='Distance_km', y='Time_taken (min)', opacity=0.3,
    trendline="ols", color_discrete_sequence=['#CB202D'],
    title=f"Distance vs Delivery Time (r = {corr:.2f})"
)
fig4.update_layout(xaxis_title="Distance (km)", yaxis_title="Delivery Time (minutes)")
st.plotly_chart(fig4, width='stretch')
r_squared_pct = corr ** 2 * 100
st.warning(
    f"⚠️ **Key finding:** distance alone explains only **~{r_squared_pct:.0f}%** (r²) of delivery time variation. "
    f"Most naive ETA formulas rely on distance — this is direct evidence why that approach falls short."
)
st.divider()

st.subheader("🧠 What actually drives delivery time? (SHAP Explainability)")
st.caption("Unlike simple correlation, SHAP shows each feature's real, direction-aware contribution to predictions, in minutes.")
shap_df_sorted = shap_df.sort_values('Avg. impact on ETA (minutes)', ascending=True).tail(10)
fig5 = px.bar(
    shap_df_sorted, x='Avg. impact on ETA (minutes)', y='Feature', orientation='h',
    color='Avg. impact on ETA (minutes)', color_continuous_scale='Reds',
    title="Top 10 Feature Drivers of Delivery Time"
)
fig5.update_layout(showlegend=False, xaxis_title="Average Impact on Prediction (minutes)")
st.plotly_chart(fig5, width='stretch')
st.info("💡 **Road_traffic_density** ranks above even **Distance_km** — confirming traffic matters more than raw distance.")
st.divider()

st.subheader("💡 Business Recommendations")
r1, r2 = st.columns(2)
with r1:
    st.markdown("""
    **🚦 Traffic-Aware Dispatch**
    Re-weight routing logic to prioritize live traffic over raw distance.

    **📦 SLA Risk Flagging**
    Automatically flag orders where a rider carries 3+ simultaneous deliveries.
    """)
with r2:
    st.markdown("""
    **🌧️ Weather-Adjusted ETAs**
    Apply automatic buffers to customer-facing ETAs during Storm/Fog conditions.

    **🏍️ Rider-Aware Matching**
    Use rider age and rating as inputs for smarter order-to-rider assignment.
    """)
