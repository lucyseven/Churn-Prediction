#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle

beha_c = ['Card_Category',]
beha_n = [ 'Months_on_book', 'Total_Relationship_Count', 
                     'Credit_Limit','Months_Inactive_12_mon', 'Contacts_Count_12_mon', 
                     'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio' ]

# Load the trained model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Add a sidebar to the app
st.sidebar.title('K-Means Model')

file1 = st.file_uploader('Upload a CSV file' ,key = 'file1')
 
    
#Check if a file was uploaded
if file1 is not None:
    # Read the contents of the file into a DataFrame
    df = pd.read_csv(file1)   


    # Use the model to make a prediction
    prediction = model.predict(df)

# Display the prediction to the user
    st.write(prediction)


# In[ ]:




