# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model = pickle.load(open("boyu_ada.pkl", 'rb'))

app = Flask(__name__)

@app.route('/analysis')
def analysis():
    return render_template("stroke.html")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method =="POST":
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        average_glucose_level = float(request.form['average_glucose_level'])
        bmi = float(request.form['bmi'])
        stroke_history = int(request.form['stroke_history'])
        stress_levels = float(request.form['stress_levels'])
        systolic_bp = int(request.form['systolic_bp'])
        diastolic_bp = int(request.form['diastolic_bp'])
        hdl = int(request.form['hdl'])
        ldl = int(request.form['ldl'])
        gender = request.form['gender']
        gender_female = 1 if gender == 'female' else 0
        gender_male = 1 if gender == 'male' else 0
        smoking_status = request.form['smoking']
        smoking_status_currently_smokes = 1 if smoking_status == 'smokes' else 0
        smoking_status_formerly_smoked = 1 if smoking_status == 'formerly smoked' else 0
        smoking_status_non_smoker = 1 if smoking_status == 'never smoked' else 0
        alcohol_intake = request.form['alcohol_intake']
        physical_activity = request.form['physical_activity']
        family_history_of_stroke = request.form['family_history_of_stroke']
        family_history_of_stroke = request.form['family_history_of_stroke']
        dietary_habits = request.form['dietary_habits']
        alcohol_intake_frequent_drinker = 1 if alcohol_intake == 'Frequent Drinker' else 0
        alcohol_intake_never = 1 if alcohol_intake == 'Never' else 0
        alcohol_intake_rarely = 1 if alcohol_intake == 'Rarely' else 0
        alcohol_intake_social_drinker = 1 if alcohol_intake == 'Social Drinker' else 0
        physical_activity_high = 1 if physical_activity == 'High' else 0
        physical_activity_low = 1 if physical_activity == 'Low' else 0
        physical_activity_moderate = 1 if physical_activity == 'Moderate' else 0
        family_history_of_stroke_no = 1 if family_history_of_stroke == 'No' else 0
        family_history_of_stroke_yes = 1 if family_history_of_stroke == 'Yes' else 0
        dietary_habits_gluten_free = 1 if dietary_habits == 'Gluten-Free' else 0
        dietary_habits_keto = 1 if dietary_habits == 'Keto' else 0
        dietary_habits_non_vegetarian = 1 if dietary_habits == 'Non-Vegetarian' else 0
        dietary_habits_paleo = 1 if dietary_habits == 'Paleo' else 0
        dietary_habits_pescatarian = 1 if dietary_habits == 'Pescatarian' else 0
        dietary_habits_vegan = 1 if dietary_habits == 'Vegan' else 0
        dietary_habits_vegetarian = 1 if dietary_habits == 'Vegetarian' else 0
        
        feature = scaler.fit_transform([[age, hypertension, heart_disease, average_glucose_level, bmi,
                                        stroke_history, stress_levels, systolic_bp, diastolic_bp, hdl, ldl,
                                        gender_female, gender_male, smoking_status_currently_smokes,
                                        smoking_status_formerly_smoked, smoking_status_non_smoker,alcohol_intake_frequent_drinker, alcohol_intake_never, alcohol_intake_rarely, alcohol_intake_social_drinker,
                                        physical_activity_high, physical_activity_low, physical_activity_moderate,
                                        family_history_of_stroke_no, family_history_of_stroke_yes,
                                        dietary_habits_gluten_free, dietary_habits_keto, dietary_habits_non_vegetarian,
                                        dietary_habits_paleo, dietary_habits_pescatarian, dietary_habits_vegan, dietary_habits_vegetarian
                                        ]])

        prediction = model.predict(feature)[0]
        prediction_text = "YES" if prediction == 1 else "NO"

        return render_template("index.html", prediction_text=f"Chance of Stroke Prediction is --> {prediction_text}")

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)