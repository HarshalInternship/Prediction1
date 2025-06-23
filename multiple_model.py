# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:43:05 2025
@author: DELL
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_diseaseLR.sav', 'rb'))

# Parkinson’s model loaded from dict
parkinsons_data = pickle.load(open('parkinsons_model.sav', 'rb'))
parkinsons_model = parkinsons_data['model'] if isinstance(parkinsons_data, dict) else parkinsons_data

# Sidebar
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ----------------------- Diabetes -----------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Pregnancies = st.text_input('Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose')
    with col3:
        BloodPressure = st.text_input('Blood Pressure')
    with col4:
        SkinThickness = st.text_input('Skin Thickness')

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        Insulin = st.text_input('Insulin')
    with col6:
        BMI = st.text_input('BMI')
    with col7:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col8:
        Age = st.text_input('Age')

    if st.button('Diabetes Test Result'):
        try:
            input_data = [[
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ]]
            prediction = diabetes_model.predict(input_data)
            st.success('Diabetic' if prediction[0] == 1 else 'Not Diabetic')
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------- Heart -----------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        cp = st.text_input('Chest Pain Type (0–3)')
        chol = st.text_input('Cholesterol')
        restecg = st.text_input('Resting ECG (0–2)')
        exang = st.text_input('Exercise Induced Angina (1/0)')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)')
        trestbps = st.text_input('Resting BP')
        fbs = st.text_input('Fasting BS > 120 (1/0)')
        thalach = st.text_input('Max Heart Rate')
        oldpeak = st.text_input('Oldpeak')
    with col3:
        slope = st.text_input('Slope (0–2)')
        ca = st.text_input('Vessels Colored (0–3)')
        thal = st.text_input('Thal (1=Normal, 2=Fixed, 3=Reversible)')

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [[
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]]
            prediction = heart_disease_model.predict(input_data)
            st.success('Has Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------- Parkinson's -----------------------
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('Average Fundamental Frequency (Fo)')
        jitter_percent = st.text_input('Jitter (%)')
        rap = st.text_input('RAP')
        shimmer = st.text_input('Shimmer')
        apq3 = st.text_input('APQ3')
        apq = st.text_input('APQ')
        nhr = st.text_input('Noise to Harmonic Ratio (NHR)')
        rpde = st.text_input('Recurrence Period Density Entropy (RPDE)')

    with col2:
        fhi = st.text_input('Max Fundamental Frequency (Fhi)')
        jitter_abs = st.text_input('Jitter (Abs)')
        ppq = st.text_input('PPQ')
        shimmer_dB = st.text_input('Shimmer (dB)')
        apq5 = st.text_input('APQ5')
        dda2 = st.text_input('DDA')
        hnr = st.text_input('Harmonic to Noise Ratio (HNR)')
        dfa = st.text_input('Detrended Fluctuation Analysis (DFA)')

    with col3:
        flo = st.text_input('Min Fundamental Frequency (Flo)')
        dda = st.text_input('DDA (Jitter)')
        spread1 = st.text_input('Spread1')
        spread2 = st.text_input('Spread2')
        d2 = st.text_input('D2')
        ppe = st.text_input('Pitch Period Entropy (PPE)')

    if st.button("Get Parkinson's Prediction"):
        try:
            input_data = [[
                float(fo), float(fhi), float(flo), float(jitter_percent), float(jitter_abs),
                float(rap), float(ppq), float(dda), float(shimmer), float(shimmer_dB),
                float(apq3), float(apq5), float(apq), float(dda2), float(nhr), float(hnr),
                float(rpde), float(dfa), float(spread1), float(spread2), float(d2), float(ppe)
            ]]
            prediction = parkinsons_model.predict(input_data)
            st.success("✅ Has Parkinson's Disease" if prediction[0] == 1 else "✅ No Parkinson's Disease")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
