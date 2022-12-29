#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle


# In[ ]:


# beha_c = ['Card_Category',]
# beha_n = [ 'Months_on_book', 'Total_Relationship_Count', 
#                      'Credit_Limit','Months_Inactive_12_mon', 'Contacts_Count_12_mon', 
#                      'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
#                     'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio' ]


# In[ ]:



# Load the trained model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[ ]:


# Add a sidebar to the app
st.sidebar.title('K-Means Model')


# In[ ]:


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


# In[3]:


if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Use the model to make predictions on the DataFrame
    predictions = model.predict(df)

    # Display the predictions
    st.write(predictions)

