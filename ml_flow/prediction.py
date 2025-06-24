import mlflow
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Step 1: Load the full dataset to fit the scaler
data = load_wine()
X_full = data.data
feature_names = data.feature_names
scaler = StandardScaler()
scaler.fit(X_full)

# Step 2: Get user input for all 13 features
print("🔍 Enter the following 13 chemical features of the wine:")
user_input = []
for feature in feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("❌ Please enter a valid number.")

user_input = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input)

# Step 3: Load the best model from MLflow Registry (choose stage or version)
mlflow.set_tracking_uri("http://127.0.0.1:8080")


# Or load by stage (make sure the model is in this stage via MLflow UI)
model_uri = "models:/WineClassifier/2"

model = mlflow.sklearn.load_model(model_uri)

# Step 4: Make prediction
prediction = model.predict(user_input_scaled)

# Step 5: Interpret and print result
predicted_index = int(prediction[0])
wine_class_labels = {
    0: "Cultivar 0",
    1: "Cultivar 1",
    2: "Cultivar 2"
}
class_name = wine_class_labels[predicted_index]
print(f"\n✅ Predicted Wine Class: {predicted_index} → {class_name}")

# Optional: Show warning message explanation
print("\n⚠️ If you saw a dependency warning (e.g. numpy version), run this to view model requirements:")
print(f"   mlflow.pyfunc.get_model_dependencies('{model_uri}')")
