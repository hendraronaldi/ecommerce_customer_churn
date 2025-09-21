import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from collections import OrderedDict

# --- Load Best Pipeline ---
try:
    pipeline = joblib.load("best_model_fs_pipeline.pkl")
except FileNotFoundError:
    st.error("Error: Could not load best_model_fs_pipeline.pkl")
    st.stop()

# Extract steps
preprocessor = pipeline.named_steps["preprocessor"]
selector = pipeline.named_steps["select"]  # SelectPercentile
model = pipeline.named_steps["model"]

# Get final feature names after preprocessing + selection
all_features = preprocessor.get_feature_names_out()
selected_mask = selector.get_support()
selected_features = [f for f, keep in zip(all_features, selected_mask) if keep]

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
def explain_prediction(input_df, customer_label="Custom Input"):
    st.subheader(f"🔎 Feature Impact (SHAP) for {customer_label}")
    try:
        # Step 1: Preprocess manually
        X_preprocessed = preprocessor.transform(input_df)

        # Step 2: Apply feature selection
        X_selected = selector.transform(X_preprocessed)

        # Step 3: Run SHAP on final model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_selected)

        # Handle binary vs multiclass
        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1]  # churn class
            base_value = explainer.expected_value[1]
        else:
            shap_values_to_use = shap_values
            base_value = explainer.expected_value

        # Waterfall for first row
        st.write("Detailed feature contribution for this customer:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_to_use[0],
                base_values=base_value,
                data=X_selected[0],
                feature_names=selected_features
            ),
            show=False
        )
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")


# --- Prediction function ---
def make_prediction(input_df, customer_label="Custom Input", show_explain=True):
    try:
        # Run full pipeline
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[:, 1]

        st.subheader(f"Prediction Result for {customer_label}")
        if prediction[0] == 1:
            st.error(f"High risk of Churn (Probability: {probability[0]:.2%})")
        else:
            st.success(f"Low risk of Churn (Probability: {probability[0]:.2%})")

        if show_explain:
            explain_prediction(input_df, customer_label)

        return probability[0]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Sidebar: Prefill or Manual Input ---
st.sidebar.header("Customer Profile")

if 'prefill_data' not in st.session_state:
    st.session_state.prefill_data = None

if st.sidebar.button("🎲 Prefill from Random Test Set"):
    try:
        test_df = pd.read_csv("test_set.csv")
        test_df = test_df.drop(columns=["Churn"], errors="ignore")
        st.session_state.prefill_data = test_df.sample(
            1, random_state=np.random.randint(0, 10000)
        ).iloc[0].to_dict()
        st.sidebar.success("Random test set row loaded!")
    except FileNotFoundError:
        st.sidebar.error("test_set.csv not found!")

# Collect user inputs
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
    user_input_dict[feature] = st.sidebar.selectbox(
        f"Select {feature}", options, 
        index=options.index(default_value) if default_value in options else 0
    )

st.sidebar.subheader("Ordinal Data")
default_city_tier = int(st.session_state.prefill_data['CityTier']) if st.session_state.prefill_data else 1
user_input_dict['CityTier'] = st.sidebar.slider("CityTier", 1, 3, default_city_tier)

default_satisfaction_score = int(st.session_state.prefill_data['SatisfactionScore']) if st.session_state.prefill_data else 3
user_input_dict['SatisfactionScore'] = st.sidebar.slider("SatisfactionScore", 1, 5, default_satisfaction_score)

# Predict Button
if st.sidebar.button("Predict Churn"):
    user_df = pd.DataFrame([user_input_dict])

    # Map binary fields
    if "Complain" in user_df:
        user_df["Complain"] = user_df["Complain"].map({"No": 0, "Yes": 1}).fillna(user_df["Complain"])
    if "Gender" in user_df:
        user_df["Gender"] = user_df["Gender"].map({"Male": "Male", "Female": "Female"})

    make_prediction(user_df, customer_label="Manual Input")
