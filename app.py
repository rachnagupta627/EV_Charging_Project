
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="EV Charging Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Dashboard title
st.title("⚡ EV Charging Demand Prediction Dashboard")

# Sidebar navigation
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Data Analysis",
        "Regression Prediction",
        "Classification Prediction",
        "Demand Forecasting",
        "Model Insights"
    ]
)

# Display selected page
st.header(page)


# Load cleaned EV charging dataset
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Data_row/cleaned data/ChargingRecords_Cleaned.csv"
    )
    df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
    return df

df = load_data()

# Load trained regression model
@st.cache_resource
def load_regression_model():
    return joblib.load("models/lightgbm_regression.pkl")

regression_model = load_regression_model()

# Load trained classification model
@st.cache_resource
def load_classification_model():
    return joblib.load("models/lightgbm_classification.pkl")

classification_model = load_classification_model()

# Check model input features
print("Regression Model Input Features:")
print(regression_model.named_steps["preprocessor"].feature_names_in_)

# Check classification model input features
print("Classification Model Input Features:")
print(classification_model.named_steps["preprocessor"].feature_names_in_)


# Reset button
if st.sidebar.button("Reset Filters"):
    st.session_state["location_filter"] = "All"
    st.session_state["charger_filter"] = "All"

# Sidebar filters
st.sidebar.subheader("Data Filters")

selected_location = st.sidebar.selectbox(
    "Select Location",
    ["All"] + sorted(df["Location"].dropna().unique().tolist()),
    key="location_filter"
)

selected_charger = st.sidebar.selectbox(
    "Select Charger Type",
    ["All"] + sorted(df["ChargerType"].dropna().unique().tolist()),
    key="charger_filter"
)

# Apply filters
if selected_location != "All":
    df = df[df["Location"] == selected_location]

if selected_charger != "All":
    df = df[df["ChargerType"] == selected_charger]


# Overview page
if page == "Overview":

    # Calculate KPIs
    total_sessions = len(df)
    total_demand = df["Demand"].sum()
    avg_demand = df["Demand"].mean()
    total_chargers = df["ChargerID"].nunique()

    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sessions", f"{total_sessions:,}")
    col2.metric("Total Demand", f"{total_demand / 1_000_000:.2f}M")
    col3.metric("Average Demand", f"{avg_demand:.2f}")
    col4.metric("Unique Chargers", f"{total_chargers:,}")

    
    # Charging demand by location
    st.subheader("Charging Demand by Location")

    location_demand = (
        df.groupby("Location")["Demand"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        location_demand,
        x_label="Charging Location",
        y_label="Total Charging Demand",
    )


    # Charging demand by charger type
    st.subheader("Charging Demand by Charger Type")

    charger_demand = (
        df.groupby("ChargerType")["Demand"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        charger_demand,
        x_label="ChargerType",
        y_label="Total Chargeing Demand",
        )

    # Daily charging demand trend
    st.subheader("Daily Charging Demand Trend")

    daily_demand = (
        df.groupby(df["StartDatetime"].dt.date)["Demand"]
        .sum()
    )

    daily_demand.index = pd.to_datetime(daily_demand.index)

    st.line_chart(
        daily_demand,
        x_label="Month/Date",
        y_label="Total Chargeing Demand"
        )

    
elif page == "Data Analysis":

    st.subheader("Charging Sessions by Hour")

    # Extract hour from charging start time
    df["StartHour"] = df["StartDatetime"].dt.hour

    # Count charging sessions for each hour
    hourly_sessions = (
        df.groupby("StartHour")
        .size()
        .reindex(range(24), fill_value=0)
    )



    # Display chart
    st.bar_chart(
        hourly_sessions,
        x_label="Hour of Day (0–23)",
        y_label="Number of Charging Sessions"
    )

        # Charging demand by weekday
    st.subheader("Charging Demand by Weekday")

    # Extract weekday name
    df["Weekday"] = df["StartDatetime"].dt.day_name()

    # Arrange weekdays in correct order
    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    # Calculate total demand by weekday
    weekday_demand = (
        df.groupby("Weekday")["Demand"]
        .sum()
        .reindex(weekday_order, fill_value=0)
    )


    # Display chart
    st.bar_chart(
        weekday_demand,
        x_label="Day of Week",
        y_label="Total Charging Demand",
        sort=False
    )

elif page == "Regression Prediction":

    st.subheader("Predict EV Charging Demand")

    location = st.selectbox(
        "Select Location",
        sorted(df["Location"].dropna().unique())
    )

    charger_type = st.selectbox(
        "Select Charger Type",
        sorted(df["ChargerType"].dropna().unique())
    )

    company = st.selectbox(
        "Select Charger Company",
        sorted(df["ChargerCompany"].dropna().unique())
    )

    start_hour = st.slider("Charging Start Hour", 0, 23, 18)
    day_of_week = st.selectbox(
        "Day of Week (0 = Monday)",
        list(range(7))
    )
    month = st.slider("Month", 1, 12, 6)

    previous_demand = st.number_input(
        "Previous Charger Demand",
        min_value=0.0,
        value=17.0
    )

    if st.button("Predict Charging Demand"):

        input_data = pd.DataFrame([{
            "ChargerType": charger_type,
            "Location": location,
            "ChargerCompany": company,
            "StartHour": start_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": int(day_of_week >= 5),
            "PreviousChargerDemand": previous_demand
        }])

        prediction = regression_model.predict(input_data)[0]

        st.success(f"Predicted Charging Demand: {prediction:.2f}")

elif page == "Classification Prediction":

    st.subheader("Predict High or Low Charging Demand")

    location = st.selectbox(
        "Select Location",
        sorted(df["Location"].dropna().unique())
    )

    charger_type = st.selectbox(
        "Select Charger Type",
        sorted(df["ChargerType"].dropna().unique())
    )

    company = st.selectbox(
        "Select Charger Company",
        sorted(df["ChargerCompany"].dropna().unique())
    )

    start_hour = st.slider("Charging Start Hour", 0, 23, 18)

    day_of_week = st.selectbox(
        "Day of Week (0 = Monday)",
        list(range(7))
    )

    month = st.slider("Month", 1, 12, 6)

    is_weekend = 1 if day_of_week >= 5 else 0

    if st.button("Predict Demand Category"):

        input_data = pd.DataFrame([{
            "ChargerType": charger_type,
            "Location": location,
            "ChargerCompany": company,
            "StartHour": start_hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend
        }])

        prediction = classification_model.predict(input_data)[0]

        if prediction == 1:
            st.success("⚡ Predicted Charging Demand: HIGH")
            st.info("High Demand means predicted demand is 14.2 or above.")
        else:
            st.warning("🔋 Predicted Charging Demand: LOW")
            st.info("Low Demand means predicted demand is below 14.2.")


elif page == "Demand Forecasting":

    st.subheader("Future EV Charging Demand Forecast")

    # Load saved forecast
    forecast_df = pd.read_csv(
        "forecasts/future_charging_demand.csv"
    )

    # Display forecast data
    st.write("Future Forecast Data")
    st.dataframe(forecast_df.head())  

    # Future charging demand chart
    st.subheader("Future Charging Demand Trend")

    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

    forecast_chart = forecast_df.set_index("Date")

    st.line_chart(
        forecast_chart["PredictedDemand"],
        x_label="Month/Date",
        y_label="Predicted Charging Demand"
    )

elif page == "Model Insights":

    import numpy as np
    import matplotlib.pyplot as plt
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance Comparison",
        "🎯 Feature Importance",
        "🧠 SHAP Explainability",
        "🔍 LIME Local Explanation"
    ])

    # ---------------------------------------------------------
    # TAB 1: Model Performance Comparison
    # ---------------------------------------------------------
    with tab1:
        st.subheader("1. Regression Models — Charging Demand Prediction")
        regression_results = pd.DataFrame({
            "Model": [
                "LightGBM (with Historical Demand)",
                "LightGBM",
                "XGBoost",
                "Gradient Boosting",
                "Decision Tree",
                "Random Forest",
                "Linear Regression"
            ],
            "MAE": [9.5494, 9.9771, 10.1632, 10.0443, 10.1560, 10.1109, 10.3957],
            "MSE": [159.9609, 170.9370, 172.6865, 173.0961, 176.7558, 177.6567, 182.6671],
            "RMSE": [12.6476, 13.0743, 13.1410, 13.1566, 13.2950, 13.3288, 13.5154],
            "R²": [0.1738, 0.1171, 0.1081, 0.1060, 0.0871, 0.0824, 0.0566]
        })
        st.dataframe(regression_results, hide_index=True, use_container_width=True)
        st.caption("Lower MAE/MSE/RMSE and higher R² indicate better regression accuracy. LightGBM with historical charger demand achieves the best performance.")

        st.divider()

        st.subheader("2. Classification Models — High vs. Normal Demand Prediction")
        classification_results = pd.DataFrame({
            "Model": [
                "LightGBM",
                "XGBoost",
                "Logistic Regression",
                "SVM",
                "Decision Tree",
                "Random Forest",
                "KNN"
            ],
            "Accuracy": [0.6436, 0.6372, 0.6171, 0.6149, 0.6158, 0.6298, 0.6163],
            "Precision": [0.6152, 0.6167, 0.5914, 0.5891, 0.5934, 0.6140, 0.6013],
            "Recall": [0.7180, 0.6756, 0.6942, 0.6942, 0.6732, 0.6478, 0.6318],
            "F1-Score": [0.6626, 0.6448, 0.6387, 0.6373, 0.6308, 0.6305, 0.6162],
            "ROC-AUC": [0.6932, 0.6839, 0.6622, 0.6607, 0.6630, 0.6756, 0.6568]
        })
        st.dataframe(classification_results, hide_index=True, use_container_width=True)
        st.caption("Higher Accuracy, Precision, Recall, F1-Score, and ROC-AUC indicate stronger classification performance. LightGBM leads across F1-Score (0.6626) and ROC-AUC (0.6932).")

        st.divider()

        st.subheader("3. Time-Series Forecasting Models")
        forecast_results = pd.read_csv(
            "forecasts/forecast_model_comparison.csv"
        )
        st.dataframe(forecast_results.round(2), hide_index=True, use_container_width=True)
        st.caption("Lower MAE and RMSE values indicate better forecasting accuracy.")

    # ---------------------------------------------------------
    # TAB 2: Feature Importance
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Feature Importance Analysis")
        st.write("Understand which features have the strongest influence on charging demand and high-demand sessions.")

        col_reg, col_clf = st.columns(2)

        with col_reg:
            st.markdown("#### Regression Model (LightGBM)")
            reg_feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in regression_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            reg_imp = pd.DataFrame({
                "Feature": reg_feat_names,
                "Importance": regression_model.named_steps["model"].feature_importances_
            }).sort_values("Importance", ascending=False).head(12)

            st.bar_chart(
                reg_imp.set_index("Feature")["Importance"],
                horizontal=True,
                x_label="Split Importance",
                y_label="Feature"
            )
            st.dataframe(reg_imp, hide_index=True, use_container_width=True)

        with col_clf:
            st.markdown("#### Classification Model (LightGBM)")
            clf_feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in classification_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            clf_imp = pd.DataFrame({
                "Feature": clf_feat_names,
                "Importance": classification_model.named_steps["model"].feature_importances_
            }).sort_values("Importance", ascending=False).head(12)

            st.bar_chart(
                clf_imp.set_index("Feature")["Importance"],
                horizontal=True,
                x_label="Split Importance",
                y_label="Feature"
            )
            st.dataframe(clf_imp, hide_index=True, use_container_width=True)

    # Helper to prepare sample data for SHAP & LIME
    @st.cache_data
    def get_explainability_sample():
        base_df = load_data().sort_values("StartDatetime").reset_index(drop=True)
        base_df["StartHour"] = base_df["StartDatetime"].dt.hour
        base_df["DayOfWeek"] = base_df["StartDatetime"].dt.dayofweek
        base_df["Month"] = base_df["StartDatetime"].dt.month
        base_df["IsWeekend"] = (base_df["DayOfWeek"] >= 5).astype(int)
        base_df["PreviousChargerDemand"] = (
            base_df.groupby("ChargerID")["Demand"]
            .shift(1)
            .fillna(base_df["Demand"].median())
        )
        return base_df.sample(300, random_state=42).reset_index(drop=True)

    sample_df = get_explainability_sample()
    reg_cols = [
        "ChargerType", "Location", "ChargerCompany",
        "StartHour", "DayOfWeek", "Month", "IsWeekend", "PreviousChargerDemand"
    ]
    clf_cols = [
        "ChargerType", "Location", "ChargerCompany",
        "StartHour", "DayOfWeek", "Month", "IsWeekend"
    ]

    # ---------------------------------------------------------
    # TAB 3: SHAP Explainability
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Explainable AI — SHAP (SHapley Additive exPlanations)")
        st.write(
            "SHAP values quantify how much each feature contributes positively or negatively "
            "to the model's prediction compared to the baseline average."
        )

        shap_model_choice = st.radio(
            "Select Model for SHAP Analysis",
            ["Regression (Charging Demand)", "Classification (High vs. Normal Demand)"],
            horizontal=True
        )

        if shap_model_choice == "Regression (Charging Demand)":
            X_trans = regression_model.named_steps["preprocessor"].transform(sample_df[reg_cols])
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in regression_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            explainer = shap.TreeExplainer(regression_model.named_steps["model"])
            shap_values = explainer.shap_values(X_trans)
        else:
            X_trans = classification_model.named_steps["preprocessor"].transform(sample_df[clf_cols])
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in classification_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            explainer = shap.TreeExplainer(classification_model.named_steps["model"])
            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        X_trans_df = pd.DataFrame(X_trans, columns=feat_names)

        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            st.markdown("#### Global Feature Impact (Mean |SHAP|)")
            mean_abs_shap = pd.DataFrame({
                "Feature": feat_names,
                "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
            }).sort_values("Mean |SHAP Value|", ascending=False).head(12)

            st.bar_chart(
                mean_abs_shap.set_index("Feature")["Mean |SHAP Value|"],
                horizontal=True,
                x_label="Mean |SHAP Value| (Impact Magnitude)",
                y_label="Feature"
            )

        with col_shap2:
            st.markdown("#### SHAP Summary Plot (Direction & Magnitude)")
            fig_shap, ax_shap = plt.subplots(figsize=(7, 5))
            shap.summary_plot(shap_values, X_trans_df, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)

    # ---------------------------------------------------------
    # TAB 4: LIME Local Explanation
    # ---------------------------------------------------------
    with tab4:
        st.subheader("Explainable AI — LIME (Local Interpretable Model-agnostic Explanations)")
        st.write(
            "LIME explains an individual charging session prediction by approximating the model "
            "locally around that specific record."
        )

        lime_col1, lime_col2 = st.columns([1, 1])
        with lime_col1:
            lime_task = st.selectbox(
                "Select Prediction Task",
                ["Regression (Predict Demand)", "Classification (High vs. Normal Demand)"]
            )
        with lime_col2:
            sample_idx = st.slider(
                "Select Sample Charging Session Index",
                min_value=0,
                max_value=len(sample_df) - 1,
                value=0
            )

        selected_row = sample_df.iloc[[sample_idx]]
        st.write("**Selected Charging Session Details:**")
        st.dataframe(
            selected_row[reg_cols + ["Demand"]],
            hide_index=True,
            use_container_width=True
        )

        if lime_task == "Regression (Predict Demand)":
            X_trans = regression_model.named_steps["preprocessor"].transform(sample_df[reg_cols])
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in regression_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            pred_val = regression_model.predict(selected_row[reg_cols])[0]
            st.info(f"**Model Predicted Demand:** {pred_val:.2f} kWh  |  **Actual Demand:** {selected_row['Demand'].values[0]:.2f} kWh")

            lime_explainer = LimeTabularExplainer(
                training_data=X_trans,
                feature_names=feat_names,
                mode="regression",
                random_state=42
            )
            lime_exp = lime_explainer.explain_instance(
                X_trans[sample_idx],
                regression_model.named_steps["model"].predict,
                num_features=8
            )
        else:
            X_trans = classification_model.named_steps["preprocessor"].transform(sample_df[clf_cols])
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            feat_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in classification_model.named_steps["preprocessor"].get_feature_names_out()
            ]
            pred_class = classification_model.predict(selected_row[clf_cols])[0]
            pred_prob = classification_model.predict_proba(selected_row[clf_cols])[0][1]
            label_str = "HIGH Demand (≥ 14.2)" if pred_class == 1 else "LOW / Normal Demand (< 14.2)"
            st.info(f"**Predicted Category:** {label_str}  |  **High-Demand Probability:** {pred_prob:.1%}")

            lime_explainer = LimeTabularExplainer(
                training_data=X_trans,
                feature_names=feat_names,
                class_names=["Normal Demand", "High Demand"],
                mode="classification",
                random_state=42
            )
            lime_exp = lime_explainer.explain_instance(
                X_trans[sample_idx],
                classification_model.named_steps["model"].predict_proba,
                num_features=8
            )

        lime_df = pd.DataFrame(
            lime_exp.as_list(),
            columns=["Feature Condition", "Contribution Weight"]
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("#### LIME Feature Contributions Chart")
            fig_lime = lime_exp.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(fig_lime)
            plt.close(fig_lime)

        with col_l2:
            st.markdown("#### LIME Contribution Breakdown")
            st.dataframe(lime_df.round(4), hide_index=True, use_container_width=True)
