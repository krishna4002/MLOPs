import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Set MLflow tracking URI (local or remote server)
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Set MLflow experiment
mlflow.set_experiment("Wine_Classificer")

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define a shared registered model name for version tracking
REGISTERED_MODEL_NAME = "WineClassifier"

# Define 5 model configurations
model_configs = [
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        "params": {"C": 1.0, "solver": "liblinear", "max_iter": 300}
    },
    {
        "model_name": "RandomForest",
        "model_class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    },
    {
        "model_name": "KNeighbors",
        "model_class": KNeighborsClassifier,
        "params": {"n_neighbors": 5, "weights": "uniform"}
    },
    {
        "model_name": "SVC",
        "model_class": SVC,
        "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}
    },
    {
        "model_name": "GradientBoosting",
        "model_class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    }
]

# Run training, logging, and versioning
for config in model_configs:
    with mlflow.start_run():
        model_name = config["model_name"]
        model_class = config["model_class"]
        model_params = config["params"]

        # Log base metadata
        mlflow.log_param("model_type", model_name)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log and register the model under the same name
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"✅ {model_name} logged and registered to '{REGISTERED_MODEL_NAME}' with metrics: Acc={acc:.4f}, F1={f1:.4f}")