import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris
import joblib
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target labels as class names instead of numbers
df['target'] = [iris.target_names[i] for i in iris.target]  # "setosa", "versicolor", ...

# REGRESSION: Predict petal length (cm) using only 3 features
X_reg = df[["sepal length (cm)", "sepal width (cm)", "petal width (cm)"]]
y_reg = df["petal length (cm)"]
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
joblib.dump(reg_model, "models/regression_model.pkl")

# CLASSIFICATION: Predict species name
X_cls = df[iris.feature_names]
y_cls = df["target"]  # Already mapped to "setosa", etc.
cls_model = LogisticRegression(max_iter=200)
cls_model.fit(X_cls, y_cls)
joblib.dump(cls_model, "models/classification_model.pkl")
