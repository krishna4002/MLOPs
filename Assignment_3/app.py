from flask import Flask, request, jsonify
import joblib
import grpc
import iris_pb2
import iris_pb2_grpc
import socket
from sklearn.datasets import load_iris
import traceback
import numpy as np
from kafka import KafkaProducer
import json

app = Flask(__name__)

# Load models
try:
    reg_model = joblib.load("models/regression_model.pkl")
    cls_model = joblib.load("models/classification_model.pkl")
    print("[INFO] Models loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load models:", e)
    raise e

iris = load_iris()

regression_fields = ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"]
classification_fields = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

def is_valid_range(field, value):
    ranges = {
        "sepal length (cm)": (4.0, 8.0),
        "sepal width (cm)": (2.0, 4.5),
        "petal length (cm)": (1.0, 7.0),
        "petal width (cm)": (1.1, 8.5)
    }
    min_val, max_val = ranges.get(field, (float('-inf'), float('inf')))
    return min_val <= value <= max_val

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_to_kafka(model_type, input_data, prediction):
    try:
        msg = {
            "model_type": model_type,
            "input_data": input_data,
            "prediction": prediction
        }
        producer.send("iris_predictions", msg)
        producer.flush()
        print(f"[KAFKA] Sent message to iris_predictions: {msg}")
    except Exception as e:
        print("[KAFKA ERROR]", e)

def send_to_grpc(model_type, input_data, prediction):
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = iris_pb2_grpc.IrisServiceStub(channel)
            response = stub.SavePrediction(
                iris_pb2.PredictionRequest(
                    model_type=model_type,
                    sepal_length=str(input_data.get("sepal length (cm)", "")),
                    sepal_width=str(input_data.get("sepal width (cm)", "")),
                    petal_length=str(input_data.get("petal length (cm)", "")),
                    petal_width=str(input_data.get("petal width (cm)", "")),
                    prediction=str(prediction)
                )
            )
        return response.message
    except Exception as e:
        print("[gRPC ERROR]", e)
        return f"gRPC error: {str(e)}"

@app.route('/')
def index():
    return jsonify({"message": "🌸 Iris AI Prediction API is alive."})

@app.route('/status')
def status():
    return jsonify({"status": "OK", "hostname": socket.gethostname()})

@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    try:
        data = request.get_json()
        for field in regression_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                value = float(data[field])
                if not is_valid_range(field, value):
                    return jsonify({"error": f"Field '{field}' out of allowed range"}), 400
            except:
                return jsonify({"error": f"Invalid number format in field: {field}"}), 400

        input_vals = [float(data[field]) for field in regression_fields]
        prediction = reg_model.predict([input_vals])[0]

        grpc_msg = send_to_grpc("regression", data, prediction)
        send_to_kafka("regression", data, float(prediction))

        return jsonify({"prediction": float(prediction), "status": grpc_msg})
    except Exception as e:
        print("[ERROR] Regression route failed:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    try:
        data = request.get_json()
        for field in classification_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                value = float(data[field])
                if not is_valid_range(field, value):
                    return jsonify({"error": f"Field '{field}' out of allowed range"}), 400
            except:
                return jsonify({"error": f"Invalid number format in field: {field}"}), 400

        input_vals = [float(data[field]) for field in classification_fields]
        prediction = cls_model.predict([input_vals])[0]

        # Fix: Handle prediction string or class index
        if isinstance(prediction, (str, np.str_)):
            label = str(prediction)
        else:
            label = iris.target_names[int(prediction)]

        grpc_msg = send_to_grpc("classification", data, label)
        send_to_kafka("classification", data, label)

        return jsonify({"prediction": label, "status": grpc_msg})
    except Exception as e:
        print("[ERROR] Classification route failed:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.errorhandler(Exception)
def global_error_handler(e):
    print("[FATAL ERROR] Uncaught exception:")
    traceback.print_exc()
    return jsonify({"error": "Fatal server error", "message": str(e)}), 500

if __name__ == '__main__':
    print(f"[INFO] Flask server running at http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000)
