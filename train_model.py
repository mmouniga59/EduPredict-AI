
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Project folder location
folder = Path(__file__).resolve().parent

# Load student dataset
data = pd.read_csv(folder / "student_data.csv")

# Input features
X = data[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_marks"
]]

# Output to predict
y = data["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Machine Learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate evaluation metrics
error = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display results
print("Model Training Completed!")
print("Mean Absolute Error (MAE):", round(error, 2))
print("R2 Score:", round(r2, 2))

# Save trained model
with open(folder / "model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
