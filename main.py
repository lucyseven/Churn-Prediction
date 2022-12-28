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


# Add a sidebar to the app
st.sidebar.title('K-Means Model')

file1 = st.file_uploader('Upload a CSV file' ,key = 'file1')

# Check if a file was uploaded
if file1 is not None:
    # Read the contents of the file into a DataFrame
    df = pd.read_csv(file1)    
    
# Define a function that takes input data and returns the predicted output
def predict(df):
    return model.predict(df)



# Use the model to make a prediction
prediction = predict(df)

# Display the prediction to the user
st.write(prediction)

