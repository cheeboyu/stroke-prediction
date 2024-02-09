# Import necessary libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
from textblob import TextBlob

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model and scaler
training_data = pd.read_csv('cleaned_stroke_prediction.csv')

# Extract relevant features for fitting the scaler
fit_data = training_data[['Age', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)',
                          'Stroke History', 'Stress Levels', 'Systolic_BP', 'Diastolic_BP', 'HDL', 'LDL',
                          'Gender_Female', 'Gender_Male', 'Smoking Status_Currently Smokes', 'Smoking Status_Formerly Smoked',
                          'Smoking Status_Non-smoker', 'Family History of Stroke_No', 'Family History of Stroke_Yes',
                          'Alcohol Intake_Frequent Drinker', 'Alcohol Intake_Never', 'Alcohol Intake_Rarely',
                          'Alcohol Intake_Social Drinker', 'Physical Activity_High', 'Physical Activity_Low',
                          'Physical Activity_Moderate', 'Dietary Habits_Gluten-Free', 'Dietary Habits_Keto',
                          'Dietary Habits_Non-Vegetarian', 'Dietary Habits_Paleo', 'Dietary Habits_Pescatarian',
                          'Dietary Habits_Vegan', 'Dietary Habits_Vegetarian']]

# Create a scaler and fit it with your training data
scaler = StandardScaler()
scaler.fit(fit_data)

# Create a Flask application instance
app = Flask(__name__)

# Function to preprocess input data


def preprocess_input(age, hypertension, heart_disease, average_glucose_level, bmi,
                     stroke_history, stress_levels, systolic_bp, diastolic_bp, hdl, ldl,
                     gender, smoking_status, alcohol_intake, physical_activity,
                     family_history_of_stroke, dietary_habits):
    # Adjust column names based on your training data
    feature_names = fit_data.columns.tolist()

    # Create a dictionary with feature names and default values (0)
    features_dict = dict.fromkeys(feature_names, 0)

    # Update the dictionary with the provided form inputs
    features_dict.update({
        'Age': float(age),
        'Hypertension': float(hypertension),
        'Heart Disease': float(heart_disease),
        'Average Glucose Level': float(average_glucose_level),
        'Body Mass Index (BMI)': float(bmi),
        'Stroke History': float(stroke_history),
        'Stress Levels': float(stress_levels),
        'Systolic_BP': float(systolic_bp),
        'Diastolic_BP': float(diastolic_bp),
        'HDL': float(hdl),
        'LDL': float(ldl),
        'Gender_Female': 1 if gender == 'female' else 0,
        'Gender_Male': 1 if gender == 'male' else 0,
        'Smoking Status_Currently Smokes': 1 if smoking_status == 'smokes' else 0,
        'Smoking Status_Formerly Smoked': 1 if smoking_status == 'formerly smoked' else 0,
        'Smoking Status_Non-smoker': 1 if smoking_status == 'never smoked' else 0,
        'Family History of Stroke_No': 1 if family_history_of_stroke == 'No' else 0,
        'Family History of Stroke_Yes': 1 if family_history_of_stroke == 'Yes' else 0,
        'Alcohol Intake_Frequent Drinker': 1 if alcohol_intake == 'Frequent Drinker' else 0,
        'Alcohol Intake_Never': 1 if alcohol_intake == 'Never' else 0,
        'Alcohol Intake_Rarely': 1 if alcohol_intake == 'Rarely' else 0,
        'Alcohol Intake_Social Drinker': 1 if alcohol_intake == 'Social Drinker' else 0,
        'Physical Activity_High': 1 if physical_activity == 'High' else 0,
        'Physical Activity_Low': 1 if physical_activity == 'Low' else 0,
        'Physical Activity_Moderate': 1 if physical_activity == 'Moderate' else 0,
        'Dietary Habits_Gluten-Free': 1 if dietary_habits == 'Gluten-Free' else 0,
        'Dietary Habits_Keto': 1 if dietary_habits == 'Keto' else 0,
        'Dietary Habits_Non-Vegetarian': 1 if dietary_habits == 'Non-Vegetarian' else 0,
        'Dietary Habits_Paleo': 1 if dietary_habits == 'Paleo' else 0,
        'Dietary Habits_Pescatarian': 1 if dietary_habits == 'Pescatarian' else 0,
        'Dietary Habits_Vegan': 1 if dietary_habits == 'Vegan' else 0,
        'Dietary Habits_Vegetarian': 1 if dietary_habits == 'Vegetarian' else 0,
    })

    # Create a DataFrame with the correct column names and a single row of data
    features_df = pd.DataFrame([features_dict], columns=feature_names)

    # Return features DataFrame
    return features_df

# Function to make stroke prediction


def predict_stroke(features):
    # Scale the input features using the scaler
    scaled_features = scaler.transform(features)

    # Make predictions using the trained model
    prediction_proba = model.predict_proba(scaled_features)

    return prediction_proba[:, 1]  # Return the probability of stroke

# Main route for the home page with form submission handling


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        try:
            # Extract form data
            numerical_features = [request.form[feature] for feature in
                                  ['age', 'hypertension', 'heart_disease', 'average_glucose_level', 'bmi',
                                   'stroke_history', 'stress_levels', 'systolic_bp', 'diastolic_bp', 'hdl', 'ldl']]
            gender = request.form['gender']
            smoking_status = request.form['smoking']
            alcohol_intake = request.form['alcohol_intake']
            physical_activity = request.form['physical_activity']
            family_history_of_stroke = request.form['family_history_of_stroke']
            dietary_habits = request.form['dietary_habits']

            # Call preprocess_input function
            preprocessed_data = preprocess_input(*numerical_features, gender, smoking_status, alcohol_intake,
                                                 physical_activity, family_history_of_stroke, dietary_habits)

            # Make predictions using the trained model
            prediction_proba = predict_stroke(preprocessed_data)

            # Determine class label
            binary_prediction = 1 if prediction_proba[0] >= 0.5 else 0

            # Determine class labels
            # Assuming class 0 is 'NO' and class 1 is 'YES'
            class_labels = ['NO', 'YES']

            # Render the prediction result on the home page
            prediction_text = f"Chance of Stroke Prediction: {prediction_proba[0]:.2%} ({class_labels[binary_prediction]})"
            return render_template("index.html", prediction_text=prediction_text)

        except Exception as e:
            print(f"An error occurred: {str(e)}")  # Print the error details
            return render_template("index.html", prediction_text=f"Error occurred during prediction: {str(e)}")

    else:
        # Render the home page with the form
        return render_template("index.html")


@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    try:
        text = request.form['text_input']
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            sentiment = 'Positive'
        elif sentiment_score < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return render_template('sentiment_analysis.html', text=text, sentiment=sentiment)
    except Exception as e:
        # Print the error for debugging
        print(f"Error occurred during sentiment analysis: {str(e)}")
        # Return an error message
        return render_template('sentiment_analysis.html', text="Error occurred during sentiment analysis.", sentiment="error")


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
