import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from collections import OrderedDict

# --- Load RFE pipeline ---
try:
    pipeline_rfe = joblib.load("best_selection_model_pipeline.pkl")
except FileNotFoundError:
    st.error("Error: Could not load best_selection_model_pipeline.pkl")
    st.stop()

# Extract preprocessor, model, and selected features
preprocessor = pipeline_rfe.named_steps["preprocessor"]
model = pipeline_rfe.named_steps["model"]
selected_features = preprocessor.get_feature_names_out()[pipeline_rfe.named_steps["feature_selection"].support_]

# --- App Title ---
st.title("📊 Customer Churn Prediction & Simulation")
st.markdown("Simulate churn likelihood, explore what-if scenarios, and get explanations.")
st.write("---")

# --- Feature categories ---
numerical_features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]
ordinal_features = ['CityTier', 'SatisfactionScore']
categorical_features = [
    'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice',
    'PreferredPaymentMode', 'Gender', 'Complain'
]

# --- SHAP explanation ---
def explain_prediction(final_input_df, customer_label="Custom Input"):
    st.subheader(f"🔎 Feature Impact (SHAP) for {customer_label}")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_input_df)

        # Waterfall (single prediction)
        st.write("Detailed feature contribution for this customer:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], final_input_df.iloc[0]
        )
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

# --- Prediction function ---
def make_prediction(input_df, customer_label="Custom Input", show_explain=True):
    try:
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_input, columns=preprocessor.get_feature_names_out())

        # Select only RFE-selected features
        final_input_df = processed_df[selected_features]

        # Prediction
        prediction = model.predict(final_input_df)
        probability = model.predict_proba(final_input_df)[:, 1]

        st.subheader(f"Prediction Result for {customer_label}")
        if prediction[0] == 1:
            st.error(f"High risk of Churn (Probability: {probability[0]:.2%})")
        else:
            st.success(f"Low risk of Churn (Probability: {probability[0]:.2%})")

        if show_explain:
            explain_prediction(final_input_df, customer_label)

        return probability[0]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Sidebar: Prefill or Manual Input ---
st.sidebar.header("Customer Profile")

# Initialize session state for prefilled data if it doesn't exist
if 'prefill_data' not in st.session_state:
    st.session_state.prefill_data = None

# Button to prefill random test set row
if st.sidebar.button("🎲 Prefill from Random Test Set"):
    try:
        test_df = pd.read_csv("test_set.csv")
        test_df = test_df.drop(columns=["Churn"], errors="ignore")  # drop label if exists
        # Store the sampled data in session state
        st.session_state.prefill_data = test_df.sample(1, random_state=np.random.randint(0, 10000)).iloc[0].to_dict()
        st.sidebar.success("Random test set row loaded!")
    except FileNotFoundError:
        st.sidebar.error("test_set.csv not found!")

# Collect user inputs using session state for default values
user_input_dict = OrderedDict()

st.sidebar.subheader("Numerical Data")
for feature in numerical_features:
    default_value = float(st.session_state.prefill_data[feature]) if st.session_state.prefill_data else 0.0
    user_input_dict[feature] = st.sidebar.number_input(f"{feature}", value=default_value)

st.sidebar.subheader("Categorical Data")
for feature in categorical_features:
    options = list(
        preprocessor.named_transformers_['nominal_cat']
        .named_steps['onehot']
        .categories_[categorical_features.index(feature)]
    )

    default_value = st.session_state.prefill_data[feature] if st.session_state.prefill_data else options[0]
    user_input_dict[feature] = st.sidebar.selectbox(f"Select {feature}", options, index=options.index(default_value) if default_value in options else 0)

st.sidebar.subheader("Ordinal Data")
default_city_tier = int(st.session_state.prefill_data['CityTier']) if st.session_state.prefill_data else 1
user_input_dict['CityTier'] = st.sidebar.slider("CityTier", 1, 3, default_city_tier)

default_satisfaction_score = int(st.session_state.prefill_data['SatisfactionScore']) if st.session_state.prefill_data else 3
user_input_dict['SatisfactionScore'] = st.sidebar.slider("SatisfactionScore", 1, 5, default_satisfaction_score)

# Predict Button
if st.sidebar.button("Predict Churn"):
    user_df = pd.DataFrame([user_input_dict])

    # Map binary to match training encoding
    if "Complain" in user_df:
        user_df["Complain"] = user_df["Complain"].map({"No": 0, "Yes": 1}).fillna(user_df["Complain"])
    if "Gender" in user_df:
        user_df["Gender"] = user_df["Gender"].map({"Male": "Male", "Female": "Female"})

    prob = make_prediction(user_df, customer_label="Manual Input")