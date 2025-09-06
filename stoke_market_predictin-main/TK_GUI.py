import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
import pickle
import os

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def process_and_predict(df):
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df1 = df['Close'].values.reshape(-1, 1)

    df1 = scaler.transform(df1)
    training_size = int(len(df1) * 0.65)
    test_data = df1[training_size:]

    time_step = 100
    _, _ = create_dataset(df1[:training_size], time_step)
    X_test, _ = create_dataset(test_data, time_step)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    x_input = test_data[len(test_data) - 100:].reshape(1, -1)
    temp_input = list(x_input[0])

    lst_output = []
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:]).reshape(1, -1, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
        else:
            x_input = np.array(temp_input).reshape(1, 100, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
        lst_output.append(yhat[0])
        i += 1

    return df1, lst_output

def animate_chart(ax, fig, df1, lst_output):
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    true_data = scaler.inverse_transform(df1[len(df1) - 100:])
    pred_data = scaler.inverse_transform(lst_output)

    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#888')
    ax.spines['top'].set_color('#888')
    ax.spines['right'].set_color('#888')
    ax.spines['left'].set_color('#888')
    ax.title.set_color('#fff')
    ax.yaxis.label.set_color('#ccc')
    ax.xaxis.label.set_color('#ccc')

    line1, = ax.plot([], [], color='#00bcd4', label='Past 100 Days', linewidth=2)
    line2, = ax.plot([], [], color='#ff9800', linestyle='--', linewidth=2, label='Predicted Next 30 Days')
    ax.set_xlim(0, 130)
    ax.set_ylim(min(true_data.min(), pred_data.min()) * 0.95,
                max(true_data.max(), pred_data.max()) * 1.05)

    ax.set_title("📈 30-Day Stock Price Forecast", fontsize=16)
    ax.legend(facecolor='#1f1f2e', edgecolor='#888', labelcolor='white')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(frame):
        if frame < 100:
            line1.set_data(day_new[:frame], true_data[:frame])
        else:
            pred_frame = frame - 100
            line2.set_data(day_pred[:pred_frame], pred_data[:pred_frame])
        return line1, line2

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 130),
        init_func=init,
        interval=30,
        repeat=False
    )
    return ani

def load_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return

    try:
        df = pd.read_csv(filepath)
        df1, prediction = process_and_predict(df)

        for widget in frame.winfo_children():
            widget.destroy()

        heading_label.config(
            text=f"📊 Here's your \"{os.path.basename(filepath)}\" analysis of next 30 days"
        )

        fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        ani = animate_chart(ax, fig, df1, prediction)
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{e}")

# -------- GUI DARK MODE SETUP ----------
root = tk.Tk()
root.title("🌚 Stylish Dark Stock Predictor")
root.geometry("1100x700")
root.configure(bg="#121212")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                foreground="#ffffff",
                background="#3f51b5",
                padding=10,
                borderwidth=0)
style.map("TButton",
          background=[("active", "#5c6bc0")],
          foreground=[("active", "#ffffff")])

heading_label = tk.Label(
    root,
    text="📂 Upload a stock CSV file to begin...",
    font=("Segoe UI", 18, "bold"),
    bg="#121212",
    fg="#eeeeee"
)
heading_label.pack(pady=20)

upload_btn = ttk.Button(root, text="📁 Upload CSV File", command=load_file)
upload_btn.pack(pady=10)

frame = tk.Frame(root, bg="#121212")
frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

root.mainloop()
