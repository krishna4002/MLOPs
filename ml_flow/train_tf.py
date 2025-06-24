import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Wine_Classification_TensorFlow")
REGISTERED_MODEL_NAME = "WineClassifier"

# Data
data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target.reshape(-1, 1)
y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)

# Model configurations
configs = [
    {"units": 16, "activation": "relu", "epochs": 50, "batch_size": 16},
    {"units": 32, "activation": "relu", "epochs": 60, "batch_size": 16},
]

# Training loop
for config in configs:
    with mlflow.start_run():
        mlflow.log_params(config)

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(13,)),
            tf.keras.layers.Dense(config["units"], activation=config["activation"]),
            tf.keras.layers.Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], verbose=0)

        # Evaluation
        preds = np.argmax(model.predict(X_test), axis=1)
        true = np.argmax(y_test, axis=1)
        acc = accuracy_score(true, preds)
        prec = precision_score(true, preds, average="macro")
        rec = recall_score(true, preds, average="macro")
        f1 = f1_score(true, preds, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1
        })

        # Log model with versioning
        mlflow.tensorflow.log_model(model, 
                                    artifact_path="model", 
                                    registered_model_name=REGISTERED_MODEL_NAME)

        print(f"TF Model logged → Acc={acc:.4f}, F1={f1:.4f}")