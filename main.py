#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

file = st.file_uploader('Upload a CSV file')
# Check if a file was uploaded
if file is not None:
    # Read the contents of the file into a DataFrame
    df = pd.read_csv(file)

    # Display the DataFrame to the user
    st.write(df)

# Define a function that takes input data and returns the predicted output
def predict(df):
    return model.predict(df)

# Add a sidebar to the app
st.sidebar.title('K-Means Model')

file = st.file_uploader('Upload a CSV file')

# Check if a file was uploaded
if file is not None:
    # Read the contents of the file into a DataFrame
    df = pd.read_csv(file)

# Use the model to make a prediction
prediction = predict(df)

# Display the prediction to the user
st.write(prediction)

