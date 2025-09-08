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
# nominal_features = ['PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice', 'PreferredPaymentMode']
ordinal_features = ['CityTier', 'SatisfactionScore']
# binary_features = ['Gender', 'Complain']
categorical_features = ['PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'Complain']

original_feature_order = numerical_features + ordinal_features + categorical_features

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

        return probability[0]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Sidebar mode ---
option = st.sidebar.radio("Choose Simulation Mode", ["Test Set", "Manual Input"])

# --- Test Set ---
if option == "Test Set":
    st.header("📉 Churn Reduction Simulation on Test Set")

    # uploaded_file = st.file_uploader("Upload Test Set CSV (with CustomerID and Churn column)", type=["csv"])
    # if uploaded_file:
    test_df = pd.read_csv('test_set.csv', index_col="CustomerID")

    st.write("Test Set Shape:", test_df.shape)
    if "Churn" in test_df.columns:
        true_churn_rate = test_df["Churn"].mean()
        st.write(f"✅ True Churn Rate in Test Set: {true_churn_rate:.2%}")

    # Sidebar for interventions
    st.sidebar.subheader("Intervention Simulation")
    inc_tenure = st.sidebar.slider("Increase Tenure (months)", 0, 12, 0)
    resolve_complain_prop = st.sidebar.slider("Reduce Complaints by (%)", 0, 100, 0)
    inc_cashback = st.sidebar.slider("Increase Cashback (%)", 0, 50, 0)
    reduce_inactivity = st.sidebar.slider("Reduce Inactivity Days", 0, 30, 0)

    if st.sidebar.button("Run Simulation"):
        sim_df = test_df.copy()

        # --- Run base prediction ---
        def run_prediction(df):
            input_df = df.drop(columns=["Churn"], errors="ignore")
            processed = preprocessor.transform(input_df)
            processed_df = pd.DataFrame(processed, columns=preprocessor.get_feature_names_out())
            final_df = processed_df[simplified_features]
            preds = simplified_model.predict(final_df)
            probs = simplified_model.predict_proba(final_df)[:, 1]
            return preds, probs

        base_preds, base_probs = run_prediction(test_df)
        sim_df_probs = pd.Series(base_probs, index=test_df.index)

        # --- Apply interventions ---
        sim_df["Tenure"] = sim_df["Tenure"] + inc_tenure
        sim_df["CashbackAmount"] = sim_df["CashbackAmount"] * (1 + inc_cashback / 100)
        sim_df["DaySinceLastOrder"] = np.maximum(0, sim_df["DaySinceLastOrder"] - reduce_inactivity)

        # Reduce complaints proportionally, only for customers with Complain==1
        resolved_count = 0
        if resolve_complain_prop > 0 and "Complain" in sim_df.columns:
            complainers = sim_df[sim_df["Complain"] == 1].index
            if len(complainers) > 0:
                num_to_reduce = int(len(complainers) * resolve_complain_prop / 100)

                # sort complainers by descending churn probability (highest risk first)
                high_risk_complainers = sim_df_probs.loc[complainers].sort_values(ascending=False).index[:num_to_reduce]

                # set their complaints to 0 (resolved)
                sim_df.loc[high_risk_complainers, "Complain"] = 0

                # track resolved complaints
                resolved_count = len(high_risk_complainers)

        # Later in results section
        if resolved_count > 0:
            st.info(f"✅ Resolved {resolved_count} complaints out of {len(complainers)} complainers "
                    f"({resolved_count/len(complainers):.1%}).")

        # --- Run simulation prediction ---
        sim_preds, sim_probs = run_prediction(sim_df)

        # Compare churn rates
        base_pred_rate = base_preds.mean()
        sim_pred_rate = sim_preds.mean()

        # --- Prediction changes breakdown ---
        st.subheader("🔄 Prediction Changes Breakdown")
        if "Churn" in test_df.columns:
            comparison_df = pd.DataFrame({
                "TrueLabel": test_df["Churn"],
                "Pred_Before": base_preds,
                "Pred_After": sim_preds
            })

            change_0to1 = ((comparison_df["Pred_Before"] == 0) & (comparison_df["Pred_After"] == 1)).sum()
            change_1to0 = ((comparison_df["Pred_Before"] == 1) & (comparison_df["Pred_After"] == 0)).sum()

            st.write(f"📈 Predictions changed **0 → 1** (non-churn → churn): {change_0to1}")
            st.write(f"📉 Predictions changed **1 → 0** (churn → retained): {change_1to0}")

        # --- Numerical feature means before & after ---
        st.subheader("📊 Numerical Feature Summary")
        num_means_before = test_df[numerical_features].mean()
        num_means_after = sim_df[numerical_features].mean()
        feature_diff = num_means_after - num_means_before
        summary_df = pd.DataFrame({
            "Mean Before": num_means_before,
            "Mean After": num_means_after,
            "Difference": feature_diff
        })
        st.dataframe(summary_df)

        # --- Simulation metrics ---
        st.subheader("📊 Simulation Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("True Churn Rate", f"{true_churn_rate:.2%}")
        col2.metric("Predicted Churn Rate (Before)", f"{base_pred_rate:.2%}")
        col3.metric("Predicted Churn Rate (After)", f"{sim_pred_rate:.2%}",
                    delta=f"{(base_pred_rate - sim_pred_rate):.2%}")

        # --- Financial Impact Simulation ---
        st.subheader("💰 Financial Impact (CAC vs CRC)")
        if "Churn" in test_df.columns:
            retained_customers = change_1to0  # churn → retained
            lost_customers = change_0to1      # non-churn → churn

            crc = 1  # baseline retention cost
            cac_range = list(range(5, 26, 5))  # CAC = 5x to 25x CRC

            st.markdown("""
            **Explanation**  
            - **CRC (Customer Retention Cost):** Cost to retain an existing customer.  
            - **CAC (Customer Acquisition Cost):** Cost to acquire a new customer, typically **5–25× higher than CRC**.  
            - **Retained customers (1→0):** Saved churns, saving you CAC per customer.  
            - **Lost customers (0→1):** Loyal customers mistakenly treated as churn, costing only CRC per customer.  
            """)

            impact_rows = []
            for cac in cac_range:
                retained_value = retained_customers * cac
                lost_value = lost_customers * crc
                net_impact = retained_value - lost_value

                impact_rows.append({
                    "CAC Multiple": f"{cac}x",
                    "Retained Customers": retained_customers,
                    "Value from Retention (Retained × CAC)": retained_value,
                    "Lost Customers": lost_customers,
                    "Cost from False Churn (Lost × CRC)": lost_value,
                    "Net Impact": net_impact
                })

            impact_df = pd.DataFrame(impact_rows)

            # Show detailed table
            st.write("### Detailed Financial Impact")
            st.dataframe(impact_df)

            # Show bar chart
            st.write("### Visual Net Impact by CAC Multiple")
            fig, ax = plt.subplots(figsize=(6,4))
            plt.bar(impact_df["CAC Multiple"], impact_df["Net Impact"])
            plt.axhline(0, color="red", linestyle="--", linewidth=1)
            plt.ylabel("Net Impact (in CRC units)")
            plt.xlabel("CAC Multiple")
            plt.title("Net Impact of Retaining vs Losing Customers")
            st.pyplot(fig)

        # --- Changed customers details ---
        comparison_full = pd.DataFrame({
            "TrueLabel": test_df.get("Churn", np.nan),
            "Pred_Before": base_preds,
            "Proba_Before": base_probs,
            "Pred_After": sim_preds,
            "Proba_After": sim_probs
        }, index=test_df.index)

        changed_df = comparison_full[comparison_full["Pred_Before"] != comparison_full["Pred_After"]]

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
    for feature in categorical_features:
        options = list(
            preprocessor.named_transformers_['nominal_cat']
            .named_steps['onehot']
            .categories_[categorical_features.index(feature)]
        )
        user_input_dict[feature] = st.sidebar.selectbox(f"Select {feature}", options)

    st.sidebar.subheader("Ordinal Data")
    user_input_dict['CityTier'] = st.sidebar.slider("CityTier", 1, 3, 1)
    user_input_dict['SatisfactionScore'] = st.sidebar.slider("SatisfactionScore", 1, 5, 3)

    if st.sidebar.button("Predict Churn"):
        user_df = pd.DataFrame([user_input_dict])
        user_df['Complain'] = user_df['Complain'].map({'No': 0, 'Yes': 1})
        user_df['NumberOfDeviceRegistered'] = user_df['NumberOfDeviceRegistered'].astype(int)
        user_df['NumberOfAddress'] = user_df['NumberOfAddress'].astype(int)

        prob = make_prediction(user_df, customer_label="Manual Input")
