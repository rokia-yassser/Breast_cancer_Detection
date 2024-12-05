import streamlit as st
import numpy as np
import pickle
import joblib
# Load the trained model

loaded_model = joblib.load(r'C:\Users\HP\Desktop\ML,DL,basics\nti_tasks.pkl')

# Define a function for prediction
def predict_cancer(texture_mean, symmetry_mean, radius_se, concave_points_se, smoothness_worst, concave_points_worst, symmetry_worst):
    
    input_data = np.array([
        texture_mean, symmetry_mean, radius_se, concave_points_se,
        smoothness_worst, concave_points_worst, symmetry_worst
    ]).reshape(1, -1)
    
    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data).max()

    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    return diagnosis, probability

# Streamlit UI components
st.title("Cancer Prediction")

st.write("This app uses a machine learning model to predict whether a tumor is benign or malignant based on input features.")

# User input
texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=10.0)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2)
radius_se = st.number_input("Radius SE", min_value=0.0, max_value=100.0, value=5.0)
concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=1.0, value=0.05)
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=1.0, value=0.2)
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.2)
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.3)

# Prediction button
if st.button("Predict"):
    diagnosis, probability = predict_cancer(texture_mean, symmetry_mean, radius_se, 
                                            concave_points_se, smoothness_worst, 
                                            concave_points_worst, symmetry_worst)
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Diagnosis: **{diagnosis}**")
    st.write(f"Prediction Confidence: **{probability:.2f}**")

print(type(loaded_model))