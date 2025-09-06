
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model   git initgit init

# Load model
model = load_model("D:\\projects\\stoke_market_predictin-main\\stoke_market_predictin-main\\models\\model.h5")

# Load scaler
with open("D:\\projects\\stoke_market_predictin-main\\stoke_market_predictin-main\\models\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def find_close_column(columns):
    """Detect column containing 'close' (case-insensitive)."""
    for col in columns:
        if "close" in col.strip().lower():
            return col
    return None


st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("📈 Stock Price Prediction App")

st.write("Upload a CSV file containing stock data (must have a column with 'Close' in its name).")

# File uploader
uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

# Number of future days input
# future_days = st.number_input("Enter number of future days to predict:", min_value=1, max_value=100, value=30)

if uploaded_file is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    # Detect Close column
    close_col = find_close_column(data.columns)

    if close_col is None:
        st.error("Could not find a 'Close' column in your dataset. Please check the file.")
    else:
        st.success(f"Detected Close column: **{close_col}**")

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # Extract Close prices
        close_data = data[[close_col]].values

        # Scale
        scaled_close = scaler.transform(close_data)

        # Use last 100 points as input
        n_steps = 100
        temp_input = list(scaled_close[-n_steps:].flatten())

        # Forecast for future_days
        lst_output = []
        i = 0
        while i < 30:
            # Always take the last n_steps values to maintain shape
            x_input = np.array(temp_input[-n_steps:]).reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)

            # Append prediction
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i += 1

        # Inverse transform predictions
        predicted_future = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        # Build day indices
        day_new = np.arange(len(close_data))
        day_pred = np.arange(len(close_data), len(close_data) + 30)

        # Plot
        st.subheader("Future Forecast (Next 30 Days)")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(day_new, scaler.inverse_transform(scaled_close), label="Actual Close Price")
        ax.plot(day_pred, predicted_future, label="Predicted Future Close Price", linestyle="--")
        ax.legend()
        ax.set_xlabel("Day Index")
        ax.set_ylabel("Stock Price")
        st.pyplot(fig)

        # Show table of predictions
        forecast_df = pd.DataFrame({
            "Day": day_pred,
            "Predicted_Close": predicted_future.flatten()
        })
        st.subheader("Future Predictions")
        st.dataframe(forecast_df)

        # Download option
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Future Predictions CSV", csv, "future_predictions.csv", "text/csv")
