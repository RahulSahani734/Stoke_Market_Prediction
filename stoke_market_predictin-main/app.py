from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import pickle
from alpha_vantage.timeseries import TimeSeries

<<<<<<< HEAD
=======

>>>>>>> 00e3d2c (Add streamlit stock prediction app)
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = "XVIBR3YK5AC6W4OL"  # Replace with your actual key

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def fetch_and_prepare_data(ticker):
    try:
        ticker = ticker.strip().upper()
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta = ts.get_daily(symbol=ticker, outputsize='full')

        df = data[['4. close']].sort_index()
        df = df.rename(columns={'4. close': 'Close'})
        df1 = df.values
        df1 = scaler.transform(df1)

        training_size = int(len(df1) * 0.65)
        test_data = df1[training_size:]

        time_step = 100
        X_test, ytest = create_dataset(test_data, time_step)
        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input[0])

        return df1, temp_input, None
    except Exception as e:
        return None, None, f"Alpha Vantage error: {e}"

def generate_plot(df1, temp_input):
    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:]).reshape(1, -1, 1)
        else:
            x_input = np.array(temp_input).reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0])
        i += 1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    plt.figure(figsize=(12, 6))
    plt.plot(day_new, scaler.inverse_transform(df1[-100:]), label='Last 100 Days')
    plt.plot(day_pred, scaler.inverse_transform(lst_output), label='Prediction')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf8')
    plt.close()
    return plot_data

@app.route('/')
def index():
    ticker = request.args.get('ticker')
    if not ticker:
        return render_template("index.html", plot_data=None, error=None)

    df1, temp_input, error = fetch_and_prepare_data(ticker)
    if error:
        return render_template("index.html", plot_data=None, error=error)

    plot_data = generate_plot(df1, temp_input)
    return render_template("index.html", plot_data=plot_data, error=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
