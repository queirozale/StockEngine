from Forecast import predictNextDays, trainModel, saveModel
import os
from keras.models import load_model
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/check_pred', methods=['POST', 'GET'])
def check_pred():
    stock_name = request.form['ticker']
    return redirect(url_for('predictions', ticker=stock_name))


@app.route('/predictions-<string:ticker>')
def predictions(ticker):
    if str(ticker) + '.h5' in os.listdir('models/'):
        model_path = os.path.join('models', ticker + '.h5')
        model = load_model(model_path)
        results = predictNextDays(model, ticker)
        print(list(results.keys()))
        return render_template('predictions.html', results=results, ticker=ticker,
        dates=list(results.keys()))
    else:
        try:
            model = trainModel(ticker)
            saveModel(ticker)
            results = predictNextDays(model, ticker)
            return render_template('predictions.html', results=results, ticker=ticker,
            dates=list(results.keys()))
        except:
            return render_template('error.html')

@app.route('/')
def index():
    return render_template('index.html')

if(__name__) == '__main__':
    app.run(debug=False)