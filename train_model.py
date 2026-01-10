
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib 

# --- SUBPART A: LOAD DATA ---
# This loads the "Textbook" (Training Data)
# If this fails, it means Training.csv is not in the 'data' folder
try:
    dataset = pd.read_csv("data/Training.csv")
    print("✅ Data Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: Training.csv not found in 'data' folder.")
    exit()

# --- SUBPART B: PREPARE DATA ---
# X = Symptoms (The questions)
# y = Prognosis (The answer key)
X = dataset.drop("prognosis", axis=1) 
y = dataset["prognosis"]

# --- SUBPART C: TRAIN MODEL ---
print("🧠 Training the AI... (This might take a moment)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- SUBPART D: SAVE THE BRAIN ---
# We save the trained model to a file so we can use it in the website later
joblib.dump(model, "disease_model.pkl")
print("🎉 Success! Model saved as 'disease_model.pkl'")
print("You are ready for Step 3!")