import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Wine_Classification_PyTorch")
REGISTERED_MODEL_NAME = "WineClassifier"

# Data
data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Model configs
configs = [
    {"hidden_size": 32, "epochs": 50, "lr": 0.01},
    {"hidden_size": 64, "epochs": 60, "lr": 0.005},
]

# Model class
class WineNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Training loop
for config in configs:
    with mlflow.start_run():
        mlflow.log_params(config)

        model = WineNet(13, config["hidden_size"])
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Training
        for epoch in range(config["epochs"]):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluation
        preds = torch.argmax(model(X_test_tensor), axis=1).detach().numpy()
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1
        })

        # Log and register model
        mlflow.pytorch.log_model(
            model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"PT Model logged → Acc={acc:.4f}, F1={f1:.4f}")