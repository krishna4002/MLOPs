import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model
MODEL_PATH = "iris_ann_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file 'iris_ann_model.h5' not found.")

model = load_model(MODEL_PATH)

# Connect to SQLite database (or create it)
conn = sqlite3.connect('iris_predictions.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sepal_length REAL,
                sepal_width REAL,
                petal_length REAL,
                petal_width REAL,
                prediction TEXT
            )''')

# Define a scaler based on typical Iris feature ranges (for demonstration)
# In practice, you should save and reuse the actual training scaler
scaler = StandardScaler()
scaler.fit([
    [5.1, 3.5, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.3, 3.3, 6.0, 2.5]
])

species = ['setosa', 'versicolor', 'virginica']

while True:
    try:
        print("\nEnter Iris flower features (type 'exit' to quit):")
        inputs = []
        for feature in ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']:
            val = input(f"{feature}: ")
            if val.lower() == 'exit':
                raise KeyboardInterrupt
            inputs.append(float(val))

        # Scale input
        input_scaled = scaler.transform([inputs])

        # Predict
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction)
        predicted_label = species[predicted_class]

        print(f"🌼 Predicted Iris species: {predicted_label}")

        # Insert into database
        c.execute('''INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, prediction)
                     VALUES (?, ?, ?, ?, ?)''', (*inputs, predicted_label))
        conn.commit()

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print(f"⚠ Error: {e}")

# Close DB connection
conn.close()