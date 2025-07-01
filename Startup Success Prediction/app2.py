import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.markdown("""
    <style>
        body {
            background-color: #fef9f4; 
        }
        .reportview-container {
            background: #fef9f4;
        }
        .sidebar .sidebar-content {
            background-color: #ffe6ea;  
        }
        .block-container {
            padding: 2rem;
        }
        .big-font {
            font-size:22px !important;
            color: #aa336a;
        }
    </style>
""", unsafe_allow_html=True)

rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
column_order = joblib.load("column_order.pkl")

st.title("STARTUP SUCCESS PREDICTOR")

st.sidebar.header("Enter Startup Info :")   # User inputs
startup_name =st.sidebar.text_input("Startup Name", "MyCoolStartup")
category =st.sidebar.selectbox("Main Category", ['Software', 'Health', 'Finance', 'Education', 'E-Commerce'])
country =st.sidebar.selectbox("Country Code", ['USA', 'IND', 'GBR', 'CAN', 'DEU'])
funding_rounds =st.sidebar.slider("Funding Rounds", 1, 20, 3)
funding_total_usd=st.sidebar.number_input("Total Funding (USD)", value=1_000_000)
founded_year=st.sidebar.slider("Founded Year", 1995, 2025, 2015)
first_fund_year=st.sidebar.slider("First Funding Year", 1995, 2025, 2016)
last_fund_year=st.sidebar.slider("Last Funding Year", 1995, 2025, 2023)

input_data = pd.DataFrame({
    'funding_total_usd': [funding_total_usd],
    'funding_rounds': [funding_rounds],
    'founded_at_year': [founded_year],
    'first_funding_at_year': [first_fund_year],
    'last_funding_at_year': [last_fund_year],
    'main_category_' + category: [1],
    'country_code_' + country: [1]
})

for col in column_order:
    if col not in input_data.columns:  # Add missing dummy columns to match model input
        input_data[col] = 0

input_data = input_data[column_order]  # Reorder columns

num_cols = ['funding_total_usd', 'funding_rounds', 'founded_at_year',
            'first_funding_at_year', 'last_funding_at_year']
input_data[num_cols] = scaler.transform(input_data[num_cols])  # scale input data

# Predict
success_prob = rf.predict_proba(input_data)[0][1]
fail_risk = xgb.predict(input_data)[0] == 0  # if predicted 0, it's risky

# Output
st.subheader(f"Prediction for: **{startup_name}**")
st.write(f"**Success Probability:** {success_prob:.2%}")
st.write("**Failure Risk (XGBoost):**", "!!! High Risk" if fail_risk else "✅ Low Risk")

if success_prob > 0.8 and not fail_risk:
    st.success("👍 Recommended for Further Investment")
elif success_prob < 0.5 or fail_risk:
    st.error("👎 Not Recommended – High Risk Detected")
else:
    st.warning("😐 Caution – Consider Deeper Validation")

with st.expander("Show Model Input Features"):
    st.dataframe(input_data)

show_features=st.radio("🔍 Want to know which features influenced the prediction?", ["No", "Yes"])

if show_features == "Yes":

    if success_prob > 0.5 and not fail_risk:          # success path  => Random Forest
        st.subheader("🌟 Top Features Driving Success (Random Forest)")
        importances = pd.Series(rf.feature_importances_, index=column_order)

    else:                                             # failure-risk path => XGBoost
        st.subheader("⚠️ Features Suggesting Failure Risk (XGBoost)")
        importances = pd.Series(xgb.feature_importances_, index=column_order)

    # --- build horizontal bar plot inline (no helper function) ---
    top5 = importances.sort_values(ascending=True).tail(5)   # smallest-to-largest for nicer barh order
    fig, ax = plt.subplots()
    ax.barh(top5.index, top5.values, color="#96c8fa")
    ax.set_xlabel("Feature Importance Score")
    ax.grid(axis='x', linestyle='--', alpha=0.3)            
    st.pyplot(fig)

