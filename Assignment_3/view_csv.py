# view_csv.py
import pandas as pd

cls_df = pd.read_csv(r'E:\MLOPs\Assignment_2\database\classification_predictions.csv')
print("\nClassification Predictions:\n", cls_df)

reg_df = pd.read_csv(r'E:\MLOPs\Assignment_2\database\regression_predictions.csv')
print("\nRegression Predictions:\n", reg_df)
