# Part 4 – `app.py`


import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")
st.markdown("Predict stock closing prices using Machine Learning")

# ----------------------------
# Load Dataset
# ----------------------------
try:
    df = pd.read_csv("data/stock_data.csv")
except FileNotFoundError:
    st.error("Dataset not found! Run data_generator.py first.")
    st.stop()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Dataset",
        "Visualizations",
        "Prediction",
        "Model Performance"
    ]
)

# ----------------------------
# Dataset
# ----------------------------
if page == "Dataset":

    st.header("Dataset Preview")

    st.dataframe(df)

    st.write("Rows :", df.shape[0])
    st.write("Columns :", df.shape[1])

# ----------------------------
# Visualization
# ----------------------------
elif page == "Visualizations":

    st.header("Stock Data Visualization")

    fig = px.line(
        df,
        x="Date",
        y="Close",
        title="Closing Price Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="Date",
        y="Volume",
        title="Trading Volume"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="Open",
        y="Close",
        title="Open vs Close Price"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Prediction
# ----------------------------
elif page == "Prediction":

    st.header("Predict Closing Price")

    model_name = st.selectbox(
        "Choose Model",
        [
            "Linear_Regression",
            "Random_Forest",
            "Gradient_Boosting",
            "best_model"
        ]
    )

    model_path = f"models/{model_name}.pkl"

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        open_price = st.number_input("Open Price", value=100.0)

        high_price = st.number_input("High Price", value=105.0)

        low_price = st.number_input("Low Price", value=98.0)

        volume = st.number_input("Volume", value=500000)

        prev_close = st.number_input("Previous Close", value=99.0)

        ma5 = st.number_input("Moving Average 5", value=100.0)

        ma10 = st.number_input("Moving Average 10", value=101.0)

        daily_return = st.number_input(
            "Daily Return",
            value=0.01,
            format="%.4f"
        )

        volatility = st.number_input(
            "Volatility",
            value=0.02,
            format="%.4f"
        )

        if st.button("Predict"):

            features = [[
                open_price,
                high_price,
                low_price,
                volume,
                prev_close,
                ma5,
                ma10,
                daily_return,
                volatility
            ]]

            prediction = model.predict(features)

            st.success(
                f"Predicted Closing Price : ₹ {prediction[0]:.2f}"
            )

            result = pd.DataFrame({
                "Predicted Close": [prediction[0]]
            })

            csv = result.to_csv(index=False)

            st.download_button(
                "Download Prediction",
                csv,
                file_name="prediction.csv",
                mime="text/csv"
            )

    else:

        st.error("Model not found! Run train_model.py first.")

# ----------------------------
# Model Performance
# ----------------------------
elif page == "Model Performance":

    st.header("Model Comparison")

    try:

        results = pd.read_csv("models/model_results.csv")

        st.dataframe(results)

        fig = px.bar(
            results,
            x="Model",
            y="R2",
            color="Model",
            title="R² Score Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.bar(
            results,
            x="Model",
            y="RMSE",
            color="Model",
            title="RMSE Comparison"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    except:

        st.warning(
            "Run train_model.py to generate model performance."
        )
