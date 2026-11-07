
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html", prediction_text=None)


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_marks = float(request.form["assignment_marks"])

    prediction = model.predict([[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks
    ]])

    final_marks = round(float(prediction[0]), 2)

    if final_marks >= 80:
        performance = "Excellent"
        message = "Outstanding academic performance!"

    elif final_marks >= 60:
        performance = "Good"
        message = "Good academic performance!"

    elif final_marks >= 40:
        performance = "Average"
        message = "Keep improving your performance!"

    else:
        performance = "Needs Improvement"
        message = "More academic support may be helpful."

    return render_template(
        "prediction.html",
        prediction_text=final_marks,
        performance=performance,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
