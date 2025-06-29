from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model from mounted directory
MODEL_PATH = "/app/model/iris_regression_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "Iris Prediction API. Use POST /predict with sepal_Length and sepal_width and petal_length data."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = pd.DataFrame([{
        "sepal_length": data["sepal_length"],
        "sepal_width": data["sepal_width"],
        "petal_length": data["petal_length"] }])
        prediction = model.predict(input_data)[0]


        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
