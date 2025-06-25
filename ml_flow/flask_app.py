from flask import Flask, request, jsonify
import mlflow
import numpy as np
import random
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === Global Setup ===
data = load_wine()
X_full = data.data
scaler = StandardScaler().fit(X_full)
feature_names = data.feature_names
target_names = data.target_names

mlflow.set_tracking_uri("http://127.0.0.1:8080")
REGISTERED_MODEL_NAME = "WineClassifier"

@app.route('/')
def welcome():
    return jsonify({
        "message": "POST 13 wine features to /predict to classify wine type"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Validate input
        input_data = request.json
        if not all(f in input_data for f in feature_names):
            return jsonify({
                "error": f"Input must contain all 13 features: {feature_names}"
            }), 400

        # Step 2: Prepare input
        input_list = [float(input_data[f]) for f in feature_names]
        input_array = np.array(input_list).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Step 3: Randomly choose alias
        rand_val = random.random()
        alias = "champion" if rand_val > 0.5 else "challenger"
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"

        # Step 4: Load and predict
        model = mlflow.pyfunc.load_model(model_uri)
        prediction = model.predict(scaled_input)
        predicted_class = int(prediction[0])
        predicted_label = target_names[predicted_class]

        return jsonify({
            "model_alias": alias,
            "random_value": round(rand_val, 3),
            "predicted_class": predicted_class,
            "predicted_label": predicted_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)