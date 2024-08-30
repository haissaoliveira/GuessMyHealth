

from flask import Flask, render_template, request 
import joblib  # Used to load the model
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

with open('logistic_model_stroke.pkl', 'rb') as file:
    LR_stroke = pickle.load(file)
    
with open('logistic_model_heart.pkl', 'rb') as file:
    LR_heart = pickle.load(file)

with open('logistic_model_diabetes.pkl', 'rb') as file:
    LR_diabetes = pickle.load(file)


@app.route('/select_condition', methods=['POST'])
def select_condition():
    # Get the selected condition from the form
    condition = request.form.get('condition')
    
    # Render the appropriate form based on the selected condition
    if condition == 'stroke':
        return render_template('stroke_form.html')
    elif condition == 'heart_disease':
        return render_template('heart_disease_form.html')
    elif condition == 'diabetes':
        return render_template('diabetes_form.html')
    else:
        return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture the data
    highbp = request.form.get('highbp')
    highchol = request.form.get('highchol')
    cholcheck = request.form.get('cholcheck')
    bmi = int(request.form.get('bmi'))
    smoker = request.form.get('smoker')
    physactivity = request.form.get('physactivity')
    fruits = request.form.get('fruits')
    veggies = request.form.get('veggies')
    hvyalcoholconsump = request.form.get('hvyalcoholconsump')
    anyhealthcare = request.form.get('anyhealthcare')
    nodocbccost = request.form.get('nodocbccost')
    genhlth = int(request.form.get('genhlth'))
    menthlth = int(request.form.get('menthlth'))
    physhlth = int(request.form.get('physhlth'))
    diffwalk = int(request.form.get('diffwalk'))
    sex = request.form.get('sex')
    age = int(request.form.get('age'))
    education = int(request.form.get('education'))
    income = int(request.form.get('income'))
   
    # Process the data (e.g., encode "yes"/"no" as 1/0)
    highbp_encoded = 1 if highbp == 'yes' else 0
    highchol_encoded = 1 if highchol == 'yes' else 0
    cholcheck_encoded = 1 if cholcheck == 'yes' else 0
    bmi_encoded = int(bmi)
    smoker_encoded = 1 if smoker == 'yes' else 0
    physactivity_encoded = 1 if physactivity == 'yes' else 0
    fruits_encoded = 1 if fruits == 'yes' else 0
    veggies_encoded = 1 if veggies == 'yes' else 0
    hvyalcoholconsump_encoded = 1 if hvyalcoholconsump == 'yes' else 0
    anyhealthcare_encoded = 1 if anyhealthcare == 'yes' else 0
    nodocbccost_encoded = 1 if nodocbccost == 'yes' else 0
    genhlth_encoded = int(genhlth)
    menthlth_encoded = int(menthlth)
    physhlth_encoded = int(physhlth)
    diffwalk_encoded = int(diffwalk)
    sex_encoded = 1 if sex == 'yes' else 0
    age_encoded = int(age) 
    education_encoded = int(education)
    income_encoded = int(income)
   
    data = {'highbp': [highbp_encoded], 'highchol':[highchol_encoded], 'cholcheck':[cholcheck_encoded],'bmi':[bmi_encoded],'smoker':[smoker_encoded],'physactivity':[physactivity_encoded], 'fruits':[fruits_encoded], 
        'veggies':[veggies_encoded], 'hvyalcoholconsump':[hvyalcoholconsump_encoded], 'anyhealthcare':[anyhealthcare_encoded], 
        'nodocbccost':[nodocbccost_encoded], 'genhlth':[genhlth_encoded],'menthlth':[menthlth_encoded], 'physhlth':[physhlth_encoded], 
        'diffwalk':[diffwalk_encoded], 'sex':[sex_encoded], 'age':[age_encoded], 'education':[education_encoded],'income':[income_encoded]
    }

    input_df = pd.DataFrame(data)

    model_prediction = LR_stroke.predict(input_df)
    
    probabilities = LR_stroke.predict_proba(input_df) 
    
    stroke_probability = probabilities[0][1]

        # Convert to a percentage
    stroke_percentage = stroke_probability * 100

        # Create a user-friendly message
    message = f"Your probability of having stroke is {stroke_percentage:.2f}%."

        # Return the message as the response
    return message

  


@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    # Capture the data
    highbp = request.form.get('highbp')
    highchol = request.form.get('highchol')
    cholcheck = request.form.get('cholcheck')
    bmi = int(request.form.get('bmi'))
    smoker = request.form.get('smoker')
    physactivity = request.form.get('physactivity')
    fruits = request.form.get('fruits')
    veggies = request.form.get('veggies')
    hvyalcoholconsump = request.form.get('hvyalcoholconsump')
    anyhealthcare = request.form.get('anyhealthcare')
    nodocbccost = request.form.get('nodocbccost')
    genhlth = int(request.form.get('genhlth'))
    menthlth = int(request.form.get('menthlth'))
    physhlth = int(request.form.get('physhlth'))
    diffwalk = int(request.form.get('diffwalk'))
    sex = request.form.get('sex')
    age = int(request.form.get('age'))
    education = int(request.form.get('education'))
    income = int(request.form.get('income'))
   
    # Process the data (e.g., encode "yes"/"no" as 1/0)
    highbp_encoded = 1 if highbp == 'yes' else 0
    highchol_encoded = 1 if highchol == 'yes' else 0
    cholcheck_encoded = 1 if cholcheck == 'yes' else 0
    bmi_encoded = int(bmi)
    smoker_encoded = 1 if smoker == 'yes' else 0
    physactivity_encoded = 1 if physactivity == 'yes' else 0
    fruits_encoded = 1 if fruits == 'yes' else 0
    veggies_encoded = 1 if veggies == 'yes' else 0
    hvyalcoholconsump_encoded = 1 if hvyalcoholconsump == 'yes' else 0
    anyhealthcare_encoded = 1 if anyhealthcare == 'yes' else 0
    nodocbccost_encoded = 1 if nodocbccost == 'yes' else 0
    genhlth_encoded = int(genhlth)
    menthlth_encoded = int(menthlth)
    physhlth_encoded = int(physhlth)
    diffwalk_encoded = int(diffwalk)
    sex_encoded = 1 if sex == 'yes' else 0
    age_encoded = int(age) 
    education_encoded = int(education)
    income_encoded = int(income)
   
    data = {'highbp': [highbp_encoded], 'highchol':[highchol_encoded], 'cholcheck':[cholcheck_encoded],'bmi':[bmi_encoded],'smoker':[smoker_encoded],'physactivity':[physactivity_encoded], 'fruits':[fruits_encoded], 
        'veggies':[veggies_encoded], 'hvyalcoholconsump':[hvyalcoholconsump_encoded], 'anyhealthcare':[anyhealthcare_encoded], 
        'nodocbccost':[nodocbccost_encoded], 'genhlth':[genhlth_encoded],'menthlth':[menthlth_encoded], 'physhlth':[physhlth_encoded], 
        'diffwalk':[diffwalk_encoded], 'sex':[sex_encoded], 'age':[age_encoded], 'education':[education_encoded],'income':[income_encoded]
    }

    input_df = pd.DataFrame(data)

    model_prediction = LR_heart.predict(input_df)

    probabilities = LR_heart.predict_proba(input_df)

    heart_probability = probabilities[0][1]

        # Convert to a percentage
    heart_percentage = heart_probability * 100

        # Create a user-friendly message
    message = f"Your probability of having heart disease attack is {heart_percentage:.2f}%."

        # Return the message as the response
    return message


@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    # Capture the data
    highbp = request.form.get('highbp')
    highchol = request.form.get('highchol')
    cholcheck = request.form.get('cholcheck')
    bmi = int(request.form.get('bmi'))
    smoker = request.form.get('smoker')
    physactivity = request.form.get('physactivity')
    fruits = request.form.get('fruits')
    veggies = request.form.get('veggies')
    hvyalcoholconsump = request.form.get('hvyalcoholconsump')
    anyhealthcare = request.form.get('anyhealthcare')
    nodocbccost = request.form.get('nodocbccost')
    genhlth = int(request.form.get('genhlth'))
    menthlth = int(request.form.get('menthlth'))
    physhlth = int(request.form.get('physhlth'))
    diffwalk = int(request.form.get('diffwalk'))
    sex = request.form.get('sex')
    age = int(request.form.get('age'))
    education = int(request.form.get('education'))
    income = int(request.form.get('income'))
   
    # Process the data (e.g., encode "yes"/"no" as 1/0)
    highbp_encoded = 1 if highbp == 'yes' else 0
    highchol_encoded = 1 if highchol == 'yes' else 0
    cholcheck_encoded = 1 if cholcheck == 'yes' else 0
    bmi_encoded = int(bmi)
    smoker_encoded = 1 if smoker == 'yes' else 0
    physactivity_encoded = 1 if physactivity == 'yes' else 0
    fruits_encoded = 1 if fruits == 'yes' else 0
    veggies_encoded = 1 if veggies == 'yes' else 0
    hvyalcoholconsump_encoded = 1 if hvyalcoholconsump == 'yes' else 0
    anyhealthcare_encoded = 1 if anyhealthcare == 'yes' else 0
    nodocbccost_encoded = 1 if nodocbccost == 'yes' else 0
    genhlth_encoded = int(genhlth)
    menthlth_encoded = int(menthlth)
    physhlth_encoded = int(physhlth)
    diffwalk_encoded = int(diffwalk)
    sex_encoded = 1 if sex == 'yes' else 0
    age_encoded = int(age) 
    education_encoded = int(education)
    income_encoded = int(income)
   
    data = {'highbp': [highbp_encoded], 'highchol':[highchol_encoded], 'cholcheck':[cholcheck_encoded],'bmi':[bmi_encoded],'smoker':[smoker_encoded],'physactivity':[physactivity_encoded], 'fruits':[fruits_encoded], 
        'veggies':[veggies_encoded], 'hvyalcoholconsump':[hvyalcoholconsump_encoded], 'anyhealthcare':[anyhealthcare_encoded], 
        'nodocbccost':[nodocbccost_encoded], 'genhlth':[genhlth_encoded],'menthlth':[menthlth_encoded], 'physhlth':[physhlth_encoded], 
        'diffwalk':[diffwalk_encoded], 'sex':[sex_encoded], 'age':[age_encoded], 'education':[education_encoded],'income':[income_encoded]
    }

    input_df = pd.DataFrame(data)

    model_prediction = LR_diabetes.predict(input_df)
    # Directly pass the input DataFrame without scaling
    probabilities = LR_diabetes.predict_proba(input_df)
    
    diabetes_probability = probabilities[0][1]

        # Convert to a percentage
    diabetes_percentage = diabetes_probability * 100

        # Create a user-friendly message
    message = f"Your probability of having diabetes is {diabetes_percentage:.2f}%."

        # Return the message as the response
    return message
    

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)



