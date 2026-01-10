import streamlit as st
import joblib
import pandas as pd

# --- PART 1: LOAD THE BRAIN ---
# Load the trained model
try:
    model = joblib.load("disease_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

# Load the symptom names from the dataset (columns) so we know what to ask
# We read just the first row to get column names
try:
    all_symptoms = pd.read_csv("data/Training.csv").columns[:-1].tolist()
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'Training.csv' is in the 'data' folder.")
    st.stop()

# --- PART 2: THE WEBSITE LAYOUT ---
st.set_page_config(page_title="Health Predictor", page_icon="🏥")
st.title("🏥 Health Prediction System (AIML)")
st.markdown("Select your symptoms below to get a preliminary diagnosis.")

# Create a dropdown menu where users can pick multiple symptoms
# This is the "User Interface" part
selected_symptoms = st.multiselect("Select Symptoms:", all_symptoms)

# --- PART 3: THE PREDICTION LOGIC ---
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        # We need to convert the user's selected words into a row of 0s and 1s
        # Create a list of zeros for all 132 symptoms
        input_data = [0] * len(all_symptoms)
        
        # Set the index to 1 for symptoms the user selected
        for symptom in selected_symptoms:
            index = all_symptoms.index(symptom)
            input_data[index] = 1
            
        # Ask the model to predict
        prediction = model.predict([input_data])
        
        # Display the result
        st.success(f"⚠️ **Predicted Diagnosis:** {prediction[0]}")
        st.info("Disclaimer: This is an AI prototype. Consult a doctor for medical advice.")




       # to run these we need python -m streamlit run app.py