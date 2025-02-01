import streamlit as st
import numpy as np
import pickle

#Load the trained model
model = pickle.load(open("model.pkl", "rb"))

#sreamlit app
st.title("Normal Model")
st.write("Predit the value based on trained model")

#User inputs
user_input = st.number_input("Your Input")

if st.button("Predict"):
    #Prepare features
    features = ([[user_input]])
    
    #Make Predictions
    prediction = model.predict(features)
    st.write(f"Predicted Value: {prediction[0]}")
