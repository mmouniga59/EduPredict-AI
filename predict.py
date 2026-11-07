
import pickle
from pathlib import Path

folder = Path(__file__).resolve().parent

# Load trained model
with open(folder / "model.pkl", "rb") as file:
    model = pickle.load(file)

print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("----------------------------------------")

study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous exam marks: "))
assignment_marks = float(input("Enter assignment marks: "))

# Prepare student data
student = [[
    study_hours,
    attendance,
    previous_marks,
    assignment_marks
]]

# Predict final marks
prediction = model.predict(student)

print("----------------------------------------")
print("Predicted Final Marks:", round(prediction[0], 2))
