
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENGINE_FILE = os.path.join(BASE_DIR, "engine_predictions.csv")
VALIDATION_FILE = os.path.join(BASE_DIR, "validation_predictions.csv")
RUL_FILE = os.path.join(BASE_DIR, "official_test_rul.csv")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    engines = pd.read_csv(ENGINE_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    official_rul = pd.read_csv(RUL_FILE)

    return engines, validation, official_rul

try:
    engines, validation, official_rul = load_data()
except Exception as error:
    st.error(f"Could not load dashboard data: {error}")
    st.info(
        "Place engine_predictions.csv, validation_predictions.csv, "
        "and official_test_rul.csv in the same folder as app.py."
    )
    st.stop()

# Normalize column names for safety.
engines["risk_label"] = engines["risk_label"].astype(str)
engines["engine_id"] = engines["engine_id"].astype(int)
engines["cycle"] = engines["cycle"].astype(int)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("✈️ Engine Analytics")
st.sidebar.caption("PySpark | Predictive Maintenance")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Engine Analysis", "Model Performance"]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: NASA C-MAPSS FD001")
st.sidebar.caption("Models: Random Forest Regression and Classification")

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------
if page == "Overview":
    st.title("Aircraft Engine Degradation Analytics")
    st.markdown(
        "### Remaining Useful Life Estimation & Failure-Risk Classification"
    )
    st.write(
        "Monitor predicted engine degradation and identify engines "
        "that may require maintenance attention."
    )

    latest = (
        engines.sort_values("cycle")
        .groupby("engine_id", as_index=False)
        .tail(1)
        .copy()
    )

    total_engines = engines["engine_id"].nunique()
    high_risk = int((latest["risk_label"] == "High").sum())
    medium_risk = int((latest["risk_label"] == "Medium").sum())
    low_risk = int((latest["risk_label"] == "Low").sum())
    average_rul = latest["predicted_rul"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines monitored", total_engines)
    c2.metric("Average predicted RUL", f"{average_rul:.1f} cycles")
    c3.metric("High-risk engines", high_risk)
    c4.metric("Medium-risk engines", medium_risk)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Latest risk classification")
        risk_counts = (
            latest["risk_label"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Engines")
        )
        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Engines",
            color="Risk",
            title="Engine count by predicted risk",
            category_orders={"Risk": ["Low", "Medium", "High"]}
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Predicted RUL distribution")
        fig = px.histogram(
            latest,
            x="predicted_rul",
            nbins=20,
            title="Latest predicted RUL across engines",
            labels={"predicted_rul": "Predicted RUL (cycles)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest engine status")
    display_cols = [
        "engine_id", "cycle", "predicted_rul", "risk_label"
    ]
    st.dataframe(
        latest[display_cols]
        .sort_values("predicted_rul")
        .rename(columns={
            "engine_id": "Engine ID",
            "cycle": "Last observed cycle",
            "predicted_rul": "Predicted RUL (cycles)",
            "risk_label": "Risk level"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk levels are model predictions, not confirmed mechanical failures. "
        "This is a simulated dataset, not a live aircraft monitoring system."
    )

# --------------------------------------------------
# ENGINE ANALYSIS PAGE
# --------------------------------------------------
elif page == "Engine Analysis":
    st.title("Individual Engine Analysis")
    st.write(
        "Explore cycle-by-cycle RUL predictions and sensor trends "
        "for an individual simulated engine."
    )

    engine_ids = sorted(engines["engine_id"].unique())
    selected_engine = st.selectbox("Select engine ID", engine_ids)

    engine_data = (
        engines[engines["engine_id"] == selected_engine]
        .sort_values("cycle")
        .copy()
    )
    latest_row = engine_data.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Engine ID", int(selected_engine))
    c2.metric("Last observed cycle", int(latest_row["cycle"]))
    c3.metric(
        "Predicted remaining useful life",
        f"{latest_row['predicted_rul']:.1f} cycles"
    )

    risk = latest_row["risk_label"]
    if risk == "High":
        st.error("Predicted risk: HIGH — prioritize review.")
    elif risk == "Medium":
        st.warning("Predicted risk: MEDIUM — monitor closely.")
    else:
        st.success("Predicted risk: LOW — continue routine monitoring.")

    fig = px.line(
        engine_data,
        x="cycle",
        y="predicted_rul",
        markers=True,
        title=f"Predicted RUL over operating cycles — Engine {selected_engine}",
        labels={
            "cycle": "Operating cycle",
            "predicted_rul": "Predicted RUL (cycles)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    sensor_cols = [
        col for col in engine_data.columns
        if col.startswith("sensor_")
        and col.split("_")[1].isdigit()
    ]

    if sensor_cols:
        selected_sensor = st.selectbox(
            "Select sensor",
            sorted(sensor_cols, key=lambda x: int(x.split("_")[1]))
        )
        fig_sensor = px.line(
            engine_data,
            x="cycle",
            y=selected_sensor,
            title=f"{selected_sensor} readings — Engine {selected_engine}",
            labels={
                "cycle": "Operating cycle",
                selected_sensor: "Sensor reading"
            }
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

    st.subheader("Engine cycle records")
    st.dataframe(
        engine_data[[
            "cycle", "predicted_rul", "risk_label"
        ] + sensor_cols[:5]],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance Evaluation")
    st.write(
        "Evaluate regression predictions on the held-out validation engines."
    )

    actual = validation["actual_rul"].to_numpy()
    predicted = validation["predicted_rul"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    c1, c2, c3 = st.columns(3)
    c1.metric("Validation MAE", f"{mae:.2f} cycles")
    c2.metric("Validation RMSE", f"{rmse:.2f} cycles")
    c3.metric("Validation R²", f"{r2:.3f}")

    st.caption(
        "These metrics use the held-out validation observations. "
        "They are not the official NASA test-set final-cycle metrics."
    )

    plot_data = validation.copy()
    fig = px.scatter(
        plot_data,
        x="actual_rul",
        y="predicted_rul",
        opacity=0.55,
        title="Actual vs predicted RUL — validation data",
        labels={
            "actual_rul": "Actual RUL (cycles)",
            "predicted_rul": "Predicted RUL (cycles)"
        }
    )

    max_value = max(
        float(plot_data["actual_rul"].max()),
        float(plot_data["predicted_rul"].max())
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="Ideal prediction",
            line={"dash": "dash"}
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction error distribution")
    plot_data["error"] = (
        plot_data["predicted_rul"] - plot_data["actual_rul"]
    )
    fig_error = px.histogram(
        plot_data,
        x="error",
        nbins=30,
        title="Prediction error (predicted RUL − actual RUL)",
        labels={"error": "Prediction error (cycles)"}
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.subheader("Validation prediction sample")
    st.dataframe(
        validation.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The classification model predicts Low, Medium, and High risk. "
        "Classification metrics should be reported separately from RUL "
        "regression metrics; regression MAE/RMSE/R² do not measure "
        "classification accuracy."
    )
