# app_streamlit.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os  # Import the os module

# Load your model
model = pickle.load(open("boyu_ada.pkl", 'rb'))

# Streamlit UI elements
st.title("Stroke Prediction")

# Form for user input
with st.form("stroke_form"):
    st.write("Fill in the details to predict stroke:")
    
    # Age
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=40.0)

    # Gender
    gender = st.radio("Gender", options=["Male", "Female"])

    # Hypertension
    hypertension = st.radio("Hypertension", options=["No", "Yes"])

    # Heart Disease
    heart_disease = st.radio("Heart Disease", options=["No", "Yes"])

    # Average Glucose Level
    average_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=80.0)

    # BMI
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    # Stroke History
    stroke_history = st.radio("Stroke History", options=["No", "Yes"])

    # Stress Levels
    stress_levels = st.number_input("Stress Levels", min_value=0.0, value=50.0)

    # Systolic Blood Pressure
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0.0, value=120.0)

    # Diastolic Blood Pressure
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, value=80.0)

    # HDL
    hdl = st.number_input("HDL", min_value=0.0, value=40.0)

    # LDL
    ldl = st.number_input("LDL", min_value=0.0, value=100.0)

    # Smoking Status
    smoking_status = st.selectbox("Smoking Status", options=["Never smoked", "Formerly smoked", "Smokes"])

    # Alcohol Intake
    alcohol_intake = st.selectbox("Alcohol Intake", options=["Never", "Rarely", "Social Drinker", "Frequent Drinker"])

    # Physical Activity
    physical_activity = st.selectbox("Physical Activity", options=["Low", "Moderate", "High"])

    # Family History of Stroke
    family_history_of_stroke = st.selectbox("Family History of Stroke", options=["No", "Yes"])

    # Dietary Habits
    dietary_habits = st.selectbox("Dietary Habits", options=["Gluten-Free", "Keto", "Non-Vegetarian", "Paleo", "Pescatarian", "Vegan", "Vegetarian"])

    # Submit button
    submitted = st.form_submit_button("Submit")

# Handle form submission
if submitted:
    # Convert categorical features to binary indicators
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    smoking_status_currently_smokes = 1 if smoking_status == 'Smokes' else 0
    smoking_status_formerly_smoked = 1 if smoking_status == 'Formerly smoked' else 0
    smoking_status_non_smoker = 1 if smoking_status == 'Never smoked' else 0

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

    # Create a DataFrame with the input values
    df = pd.DataFrame({
        "Age": [age],
        "Hypertension": [1] if hypertension == 'Yes' else [0],
        "Heart Disease": [1] if heart_disease == 'Yes' else [0],
        "Average Glucose Level": [average_glucose_level],
        "BMI": [bmi],
        "Stroke History": [1] if stroke_history == 'Yes' else [0],
        "Stress Levels": [stress_levels],
        "Systolic Blood Pressure": [systolic_bp],
        "Diastolic Blood Pressure": [diastolic_bp],
        "HDL": [hdl],
        "LDL": [ldl],
        "Gender_Female": [gender_female],
        "Gender_Male": [gender_male],
        "Smoking Status_Currently Smokes": [smoking_status_currently_smokes],
        "Smoking Status_Formerly Smoked": [smoking_status_formerly_smoked],
        "Smoking Status_Non-Smoker": [smoking_status_non_smoker],  # Check this column name
        "Alcohol Intake_Frequent Drinker": [alcohol_intake_frequent_drinker],
        "Alcohol Intake_Never": [alcohol_intake_never],
        "Alcohol Intake_Rarely": [alcohol_intake_rarely],
        "Alcohol Intake_Social Drinker": [alcohol_intake_social_drinker],
        "Physical Activity_High": [physical_activity_high],
        "Physical Activity_Low": [physical_activity_low],
        "Physical Activity_Moderate": [physical_activity_moderate],
        "Family History of Stroke_No": [family_history_of_stroke_no],
        "Family History of Stroke_Yes": [family_history_of_stroke_yes],
        "Dietary Habits_Gluten-Free": [dietary_habits_gluten_free],
        "Dietary Habits_Keto": [dietary_habits_keto],
        "Dietary Habits_Non-Vegetarian": [dietary_habits_non_vegetarian],
        "Dietary Habits_Paleo": [dietary_habits_paleo],
        "Dietary Habits_Pescatarian": [dietary_habits_pescatarian],
        "Dietary Habits_Vegan": [dietary_habits_vegan],
        "Dietary Habits_Vegetarian": [dietary_habits_vegetarian]
    })

 # Load the scaler if available, otherwise create a new one and save it
    scaler_filename = "boyu_scaler.pkl"
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if os.path.exists(scaler_filename):
        with open(scaler_filename, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    else:
        scaler = StandardScaler()

    # Check feature names in the scaler
    scaler_feature_names = set(scaler.get_feature_names_out())
    required_feature_names = set(df.columns)

    # Identify missing feature names
    missing_feature_names = required_feature_names - scaler_feature_names

    # Add missing feature names to the scaler
    if missing_feature_names:
        for feature_name in missing_feature_names:
            scaler_feature_names.add(feature_name)
        scaler = StandardScaler()
        scaler.fit(df[numeric_columns])
        with open(scaler_filename, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

    # Feature scaling (only on numeric columns)
    scaled_numeric = scaler.transform(df[numeric_columns])

    # Create a DataFrame with the scaled numeric values
    df_scaled = pd.DataFrame(scaled_numeric, columns=numeric_columns)

    # Concatenate the scaled numeric columns with the categorical columns
    df = pd.concat([df.drop(columns=numeric_columns), df_scaled], axis=1)

    # Rename columns to match the ones used during training
    column_mapping = {
        "BMI": "Body Mass Index (BMI)",
        "Diastolic Blood Pressure": "Diastolic_BP",
        "Smoking Status_Non-Smoker": "Smoking Status_Non-smoker",
        "Systolic Blood Pressure": "Systolic_BP",
        "Hypertension": "Hypertension",
        "Heart Disease": "Heart Disease",
        "Stroke History": "Stroke History"
    }

    df = df.rename(columns=column_mapping)

    # Make a prediction using the model
    prediction = model.predict(df)[0]
    prediction_text = "YES" if prediction == 1 else "NO"

    # Display the result
    st.write(f"Chance of Stroke Prediction is --> {prediction_text}")