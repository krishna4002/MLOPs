# rest_api/app.py
from flask import Flask, request, jsonify
import joblib
import grpc
import iris_pb2
import iris_pb2_grpc
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.getcwd(), 'models')
reg_model = joblib.load(os.path.join(MODEL_DIR, 'regression_model.pkl'))
cls_model = joblib.load(os.path.join(MODEL_DIR, 'classification_model.pkl'))

def send_to_grpc(model_type, input_data, prediction):
    channel = grpc.insecure_channel('localhost:50051')
    stub = iris_pb2_grpc.IrisServiceStub(channel)
    request = iris_pb2.PredictionRequest(
        model_type=model_type,
        input_data=str(input_data),
        prediction=str(prediction)
    )
    response = stub.SavePrediction(request)
    return response.message

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    data = request.json
    required_features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

    # Validate input
    if not all(feature in data for feature in required_features):
        return jsonify({'error': 'Missing required input features'}), 400

    try:
        # Extract features in order and convert to float
        features = [float(data[feature]) for feature in required_features]

        # Predict class label (string, e.g. "setosa")
        prediction_label = cls_model.predict([features])[0]

        # Send prediction to gRPC server
        message = send_to_grpc("classification", data, prediction_label)

        return jsonify({'prediction': prediction_label, 'status': message})

    except Exception as e:
        print("Error in classification prediction:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    data = request.json
    required_features = ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"]

    if not all(feature in data for feature in required_features):
        return jsonify({'error': 'Missing required input features'}), 400

    try:
        features = [float(data[feature]) for feature in required_features]
        prediction = float(reg_model.predict([features])[0])

        message = send_to_grpc("regression", data, prediction)
        return jsonify({'prediction': prediction, 'status': message})

    except Exception as e:
        print("Error in regression prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
