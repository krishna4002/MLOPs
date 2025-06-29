import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load the iris dataset
df = pd.read_csv('iris.csv')

# Prepare data using iloc as specified
X = df.iloc[:, :3]  # First 3 columns
y = df.iloc[:, 3]   # Last column (petal width)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# Initialize and train model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model R-squared: {accuracy:.4f}")

#save model
joblib.dump(model, 'iris_regression_model.pkl')
print("✅ Model saved as iris_regression_model.pkl")


# Example of model usage:
# loaded_model = joblib.load('iris_regression_model.pkl')
# prediction = loaded_model.predict([[5.1, 3.5, 1.4]])
# print(f"Predicted petal width: {prediction[0]:.1f} cm")