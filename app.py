import os
from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def create_sliding_window(data, window_size, step):
    data = (data - np.mean(data)) / np.std(data)
    X = []
    y = []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = df['Close']
        window_size = 10
        step = 1
        X, y = create_sliding_window(df, window_size, step)
        X = np.expand_dims(X, axis=2)
        train_size = int(X.shape[0] * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = Sequential()
        model.add(LSTM(50, input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
        model.save("stock_predictor.h5")
        model = load_model("stock_predictor.h5")
        predictions = model.predict(X_test)
        plt.plot(y_test, label='Ground truth')
        plt.plot(predictions, label='Predictions')
        plt.legend()
        #plt.show()
        plt.savefig("prediction_chart.png")
        return send_file("prediction_chart.png", mimetype='image/png')
        return jsonify({'predictions': str(predictions.tolist())})


if __name__ == '__main__':
    app.run(debug=True)
