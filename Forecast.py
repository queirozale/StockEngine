from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import json
import os

def getFeatures(df, index):
    last_7days = df[index-8:index]['Close'].tolist()
    return last_7days

def standardScaler(x):
    xMin, xMax = np.min(x), np.max(x)
    return [(k - xMin)/(xMax - xMin) for k in x], [xMin, xMax]

def trainModel(stock_name):
    """LSTM network Forecast Model
    input: str
        Stock name according to market symbols
    output: json
        The model trained based on the stock historical data
    """
    # Parameters of the Network
    NEURONS = 4
    BATCH_SIZE = 1
    NB_EPOCH = 5
    # DEV_SIZE = 64
    DT_SIZE = 720

    df = yf.Ticker(stock_name).history(period="25mo").reset_index()
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())

    if len(df) < DT_SIZE:
        DT_SIZE = len(df) - 8

    data_cols = []
    for i in range(1, 8):
        data_cols.append('last{}day'.format(8-i))
    data_cols.append('target')

    data = []
    for i in range(DT_SIZE):
        index = len(df) - 1
        x = getFeatures(df, index-i)
        z = standardScaler(x)
        data.append(z[0])

    data = np.array(data)
    dataModel = pd.DataFrame(data=data, columns=data_cols)

    X, y = dataModel[data_cols[:-1]].values, dataModel[data_cols[-1:]].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # X_train, X_dev, y_train, y_dev = X[:len(X) - DEV_SIZE], X[-DEV_SIZE:], y[:len(X) - DEV_SIZE], y[-DEV_SIZE:]

    model = Sequential()
    model.add(LSTM(NEURONS, batch_input_shape=(BATCH_SIZE, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    for i in range(NB_EPOCH):
        model.fit(X, y, epochs=1, batch_size=BATCH_SIZE, verbose=0, shuffle=False)
        model.reset_states()
        print("Epoch {} completed!".format(i+1))
        
    return model

def saveModel(stock_name):
    """ Save the model trained
    """
    try:
        model = trainModel(stock_name)
        model.save(os.path.join('models', stock_name + '.h5'))
    except:
        print(
        "Error in Machine Learning process. /n Try to check if you typed the right stock code.")


def predictNextDays(model, stock_name):
    """ Prediction of the next 3 days of that stock value (including today)
    input: model: model, stock_name: str
    output: results: dict
    """
    df = yf.Ticker(stock_name).history(period="1mo").reset_index()
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())

    next_dates, preds = [], []
    dt = date.today()
    c = 0
    xStart = getFeatures(df, len(df))[1:]
    while c < 3:
        z = standardScaler(xStart)
        x = np.array([z[0]])
        x = x.reshape(x.shape[0], 1, x.shape[1])
        yPred = model.predict(x)
        minValue, maxValue = z[1][0], z[1][1]
        yPred = yPred*(maxValue - minValue) + minValue
        xStart = xStart[1:] + yPred[0].tolist()
        preds.append(round(yPred[0][0], 2))
        dt = date.today() + timedelta(c)
        dt = dt.strftime('%Y-%m-%d')
        next_dates.append(dt)
        c += 1

    results = {}
    for i in range(len(next_dates)):
        results[next_dates[i]] = preds[i]

    return results

if __name__ == "__main__":
    saveModel("BTC-USD")

