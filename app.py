import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
import shap
import matplotlib.pyplot as plt

# --- Load model, preprocessor, features ---
try:
    simplified_model = joblib.load('simplified_best_model.pkl')
    simplified_features = joblib.load('simplified_features.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Error: Required files (model, preprocessor, or feature list) not found.")
    st.stop()

# --- App Title ---
st.title('📊 Customer Churn Prediction & Simulation')
st.markdown("Simulate churn likelihood, explore what-if scenarios, and get recommendations.")
st.write("---")

# --- Feature categories ---
numerical_features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]
nominal_features = ['PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice', 'PreferredPaymentMode']
ordinal_features = ['CityTier', 'SatisfactionScore']
binary_features = ['Gender', 'Complain']

original_feature_order = numerical_features + nominal_features + ordinal_features + binary_features

# --- SHAP explanation ---
def explain_prediction(final_input_df, customer_label="Custom Input"):
    st.subheader(f"🔎 Feature Impact (SHAP) for {customer_label}")
    try:
        explainer = shap.TreeExplainer(simplified_model)
        shap_values = explainer.shap_values(final_input_df)

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, final_input_df, plot_type="bar", show=False)
        st.pyplot(fig)

        st.write("Detailed feature contribution for this customer:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], final_input_df.iloc[0]
        )
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

# --- Recommendation engine ---
def generate_recommendations(raw_input):
    recs = []
    if raw_input.get("Complain", 0) == 1:
        recs.append("🚩 Customer has complained recently — resolve issues quickly.")
    if raw_input.get("DaySinceLastOrder", 0) > 60:
        recs.append("⚠️ Customer inactive — consider sending re-engagement offers.")
    if raw_input.get("CouponUsed", 0) == 0:
        recs.append("💡 No coupons used — offering a discount may help retention.")
    if raw_input.get("SatisfactionScore", 3) <= 2:
        recs.append("❗ Low satisfaction score — reach out for feedback or offer perks.")
    if raw_input.get("Tenure", 0) < 6:
        recs.append("📉 New customer with low tenure — focus on onboarding experience.")
    if not recs:
        recs.append("✅ No immediate risks detected. Maintain engagement.")
    return recs

# --- Prediction function ---
def make_prediction(input_df, customer_label="Custom Input", show_explain=True):
    try:
        input_df = input_df[original_feature_order]

        processed_input = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_input, columns=preprocessor.get_feature_names_out())
        final_input_df = processed_df[simplified_features]

        prediction = simplified_model.predict(final_input_df)
        probability = simplified_model.predict_proba(final_input_df)[:, 1]

        st.subheader(f"Prediction Result for {customer_label}")
        if prediction[0] == 1:
            st.error(f"High risk of Churn (Probability: {probability[0]:.2%})")
        else:
            st.success(f"Low risk of Churn (Probability: {probability[0]:.2%})")

        if show_explain:
            explain_prediction(final_input_df, customer_label)

        # Recommendations
        st.subheader("📌 Recommendations")
        for rec in generate_recommendations(input_df.iloc[0].to_dict()):
            st.write(rec)

        return probability[0]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Sidebar mode ---
option = st.sidebar.radio("Choose Simulation Mode", ["Upload Test Set", "Manual Input"])

# --- Upload Test Set ---
if option == "Upload Test Set":
    # uploaded_file = st.sidebar.file_uploader("Upload Test Set CSV", type=["csv"])
    # if uploaded_file:
    #     test_df = pd.read_csv(uploaded_file, index_col="CustomerID")
    #     st.sidebar.success("File uploaded successfully.")
    #     selected_id = st.sidebar.selectbox("Select CustomerID", test_df.index.tolist())
    #     if st.sidebar.button("Predict for Selected Customer"):
    #         customer_data = test_df.loc[[selected_id]]
    #         make_prediction(customer_data, customer_label=f"CustomerID {selected_id}")
    # --- New Section: Bulk Simulation on Test Set ---
    st.header("📉 Churn Reduction Simulation on Test Set")

    uploaded_file = st.file_uploader("Upload Test Set CSV (with CustomerID and Churn column)", type=["csv"])
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file, index_col="CustomerID")

        # Show quick stats
        st.write("Test Set Shape:", test_df.shape)
        if "Churn" in test_df.columns:
            true_churn_rate = test_df["Churn"].mean()
            st.write(f"✅ True Churn Rate in Test Set: {true_churn_rate:.2%}")

        # Sidebar for interventions
        st.sidebar.subheader("Intervention Simulation")
        inc_tenure = st.sidebar.slider("Increase Tenure (months)", 0, 12, 0)
        resolve_complaints = st.sidebar.checkbox("Resolve All Complaints (set to No)")
        inc_cashback = st.sidebar.slider("Increase Cashback (%)", 0, 50, 0)
        reduce_inactivity = st.sidebar.slider("Reduce Inactivity Days", 0, 30, 0)

        if st.sidebar.button("Run Simulation"):
            sim_df = test_df.copy()

            # Apply interventions
            sim_df["Tenure"] = sim_df["Tenure"] + inc_tenure
            if resolve_complaints and "Complain" in sim_df.columns:
                sim_df["Complain"] = 0  # 0 = No
            sim_df["CashbackAmount"] = sim_df["CashbackAmount"] * (1 + inc_cashback / 100)
            sim_df["DaySinceLastOrder"] = np.maximum(0, sim_df["DaySinceLastOrder"] - reduce_inactivity)

            # --- Preprocess both original & simulated ---
            def run_prediction(df):
                input_df = df.drop(columns=["Churn"], errors="ignore")  # remove true label if exists
                processed = preprocessor.transform(input_df)
                processed_df = pd.DataFrame(processed, columns=preprocessor.get_feature_names_out())
                final_df = processed_df[simplified_features]
                preds = simplified_model.predict(final_df)
                probs = simplified_model.predict_proba(final_df)[:, 1]
                return preds, probs

            # Base predictions
            base_preds, base_probs = run_prediction(test_df)
            sim_preds, sim_probs = run_prediction(sim_df)

            # Compare churn rates
            base_pred_rate = base_preds.mean()
            sim_pred_rate = sim_preds.mean()

            st.subheader("📊 Simulation Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("True Churn Rate", f"{true_churn_rate:.2%}")
            col2.metric("Predicted Churn Rate (Before)", f"{base_pred_rate:.2%}")
            col3.metric("Predicted Churn Rate (After)", f"{sim_pred_rate:.2%}",
                        delta=f"{(base_pred_rate - sim_pred_rate):.2%}")

            # Show comparison table with only changes
            comparison_df = pd.DataFrame({
                "TrueLabel": test_df.get("Churn", np.nan),
                "Pred_Before": base_preds,
                "Proba_Before": base_probs,
                "Pred_After": sim_preds,
                "Proba_After": sim_probs
            }, index=test_df.index)

            # Keep only rows where prediction or probability changed
            changed_df = comparison_df[
                (comparison_df["Pred_Before"] != comparison_df["Pred_After"])
            ]

            st.write("### Customers Affected by Simulation")
            if changed_df.empty:
                st.info("No customers were affected by the simulation (no prediction changes).")
            else:
                st.dataframe(changed_df)


# --- Manual Input ---
elif option == "Manual Input":
    st.sidebar.header("Customer Profile")
    user_input_dict = OrderedDict()

    st.sidebar.subheader("Numerical Data")
    for feature in numerical_features:
        user_input_dict[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    st.sidebar.subheader("Nominal Data")
    for feature in nominal_features:
        options = list(
            preprocessor.named_transformers_['nominal_cat']
            .named_steps['onehot']
            .categories_[nominal_features.index(feature)]
        )
        user_input_dict[feature] = st.sidebar.selectbox(f"Select {feature}", options)

    st.sidebar.subheader("Ordinal Data")
    user_input_dict['CityTier'] = st.sidebar.slider("CityTier", 1, 3, 1)
    user_input_dict['SatisfactionScore'] = st.sidebar.slider("SatisfactionScore", 1, 5, 3)

    st.sidebar.subheader("Binary Data")
    user_input_dict['Gender'] = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    user_input_dict['Complain'] = st.sidebar.selectbox("Complain", ['No', 'Yes'])

    if st.sidebar.button("Predict Churn"):
        user_df = pd.DataFrame([user_input_dict])
        user_df['Complain'] = user_df['Complain'].map({'No': 0, 'Yes': 1})
        user_df['NumberOfDeviceRegistered'] = user_df['NumberOfDeviceRegistered'].astype(int)
        user_df['NumberOfAddress'] = user_df['NumberOfAddress'].astype(int)

        prob = make_prediction(user_df, customer_label="Manual Input")

        # What-if Simulation
        st.subheader("🎯 What-if Simulation")
        st.markdown("Adjust key drivers to see how churn probability changes.")

        sim_input = user_df.copy()
        sim_input['Complain'] = st.selectbox("Simulate Complaint Resolution", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        sim_input['CouponUsed'] = st.number_input("Simulate Coupon Usage", value=int(user_df['CouponUsed'][0]), min_value=0)
        sim_input['DaySinceLastOrder'] = st.slider("Simulate Days Since Last Order", 0, 365, int(user_df['DaySinceLastOrder'][0]))
        sim_input['SatisfactionScore'] = st.slider("Simulate Satisfaction Score", 1, 5, int(user_df['SatisfactionScore'][0]))

        if st.button("Run Simulation"):
            make_prediction(sim_input, customer_label="Simulated Scenario", show_explain=False)
