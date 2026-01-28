import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PART 1: LOAD THE BRAIN ---
try:
    model = joblib.load("disease_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

try:
    all_symptoms = pd.read_csv("data/Training.csv").columns[:-1].tolist()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'Training.csv' is in the 'data' folder.")
    st.stop()

# --- PART 2: DEFINE THE RULES (HOSPITAL VS CLINIC) ---
# List of diseases that are SERIOUS (Go to Hospital)
hospital_cases = [
    "Heart Attack", "Paralysis (brain hemorrhage)", "Heart Disease", 
    "Tuberculosis", "Typhoid", "Dengue", "Hepatitis B", "Pneumonia",
    "Diabetes", "Hypoglycemia"
]

# List of diseases that are MILD (Go to Clinic)
clinic_cases = [
    "Fungal infection", "Acne", "Common Cold", "Allergy", 
    "Migraine", "Gastroenteritis", "Jaundice", "Drug Reaction"
]

# --- PART 3: THE WEBSITE LAYOUT ---
st.set_page_config(page_title="Rural Health Triage", page_icon="🏥", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏥 Rural Smart Health System</h1>", unsafe_allow_html=True)
st.markdown("### AI-Driven Triage: Determining Severity for Rural Patients")
st.write("---")

# User Interface for Symptoms
selected_symptoms = st.multiselect("Select Symptoms (Type to search):", all_symptoms)

# --- PART 4: THE PREDICTION LOGIC ---
if st.button("Analyze Condition"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # Prepare input data (0s and 1s)
        input_data = [0] * len(all_symptoms)
        for symptom in selected_symptoms:
            index = all_symptoms.index(symptom)
            input_data[index] = 1
            
        # 1. PREDICT DISEASE
        prediction = model.predict([input_data])[0]
        
        # 2. CALCULATE CONFIDENCE (RISK PROBABILITY)
        # The model gives a probability for every disease. We take the highest one.
        probability = model.predict_proba([input_data]).max() * 100
        
        st.write("---")
        st.subheader("📊 Analysis Report")
        
        # Display the Speedometer (Progress Bar)
        st.write(f"**AI Confidence Level:** {probability:.1f}%")
        if probability < 50:
            st.progress(int(probability))
            st.caption("⚠️ Low confidence. Results might be inaccurate. Please check symptoms again.")
        else:
            st.progress(int(probability))
            
        # 3. TRIAGE LOGIC (Hospital vs. Clinic)
        st.write(f"**Predicted Diagnosis:** {prediction}")
        
        if prediction in hospital_cases:
            # SERIOUS CASE
            st.error("🚨 SEVERITY: HIGH (SERIOUS)")
            st.markdown(f"""
                <div style='background-color: #ffe6e6; padding: 15px; border-radius: 10px; border: 2px solid red;'>
                    <h3 style='color: #cc0000; margin-top: 0;'>🛑 ACTION REQUIRED: GO TO HOSPITAL</h3>
                    <p>The predicted condition <b>({prediction})</b> is classified as <b>SERIOUS</b>.</p>
                    <ul>
                        <li><b>Do not</b> visit a local clinic.</li>
                        <li>Proceed immediately to the nearest District Hospital.</li>
                        <li>This condition requires advanced facilities.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        elif prediction in clinic_cases:
            # MILD CASE
            st.success("🟢 SEVERITY: LOW (MILD)")
            st.markdown(f"""
                <div style='background-color: #e6fffa; padding: 15px; border-radius: 10px; border: 2px solid green;'>
                    <h3 style='color: #006644; margin-top: 0;'>✅ ACTION REQUIRED: VISIT LOCAL CLINIC</h3>
                    <p>The predicted condition <b>({prediction})</b> is typically <b>NOT LIFE THREATENING</b>.</p>
                    <ul>
                        <li>Visit your nearest Primary Health Center (PHC) or Local Clinic.</li>
                        <li>General medication is likely sufficient.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        else:
            # UNCERTAIN / MODERATE CASE
            st.warning("⚠️ SEVERITY: MODERATE")
            st.write("Please consult a general physician for further advice.")

        st.info("ℹ️ Note: This AI is for decision support only. Always follow medical advice.")