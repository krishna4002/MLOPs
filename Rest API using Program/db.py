import sqlite3
import time
from datetime import datetime

def display_predictions():
    conn = sqlite3.connect('iris_predictions.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()

    print("ID | Sepal L | Sepal W | Petal L | Petal W | Predicted Class     | Timestamp")
    print("-" * 80)
    for row in rows:
        if len(row) == 7:  # Ensure row has all expected fields
            print("{:<3} {:<8} {:<8} {:<8} {:<8} {:<20} {}".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            ))
        else:
            print(f"Skipping malformed row: {row}")

    conn.close()

# Continuously update every 90 seconds
while True:
    display_predictions()
    print("\nWaiting 30 seconds to refresh...\n")
    time.sleep(30)