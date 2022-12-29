#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle


# In[7]:


import sklearn
print(sklearn.__version__)


# In[ ]:



# Load the trained model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[8]:


#read the template
template = pd.read_csv("BankChurners.csv")


# In[ ]:


# Add a sidebar to the app
st.sidebar.title('K-Means Model')

#add a title
st.title("Bank of Offer Customer Segmentation")


# In[ ]:


beha_c = ['Card_Category',]
beha_n = [ 'Months_on_book', 'Total_Relationship_Count', 
                     'Credit_Limit','Months_Inactive_12_mon', 'Contacts_Count_12_mon', 
                     'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio' ]


# In[ ]:



# Display a radio button group
option = st.radio("Choose an option:", ["Upload a csv file", "Enter data"])

        


# In[ ]:



if option == "Upload a  csv file":
    # Allow the user to upload a file
    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)

        # Use the model to make predictions on the DataFrame
        predictions = model.predict(df)

        # Display the predictions
        st.write(predictions)


# In[ ]:


elif option == "Enter data":
    # Allow the user to enter data
    
    #Card_Category
    card = st.selectbox('Card Category',template['Card_Category'].unique())
    
    #month on book
    month_on_book = st.number_input('Months_on_book')
    
    #total relationship
    relationship = st.number_input('Total_Relationship_Count')
    
    #credit limit 
    limit = st.number_input('Credit_Limit')
    
    #month inactive
    
    month_inactive = st.number_input('Months_Inactive_12_mon')  
        
    #contacts counts
    
    contacts = st.number_input('Contacts_Count_12_mon')
    
    #total revolving bal
    
    balance = st.number_input('Total_Revolving_Bal')
    
    #total amount change 4 to 1
    
    amount_c41 = st.number_input('Total_Amt_Chng_Q4_Q1')
    #total transaction amount
    amount = st.number_input('Total_Trans_Amt')
    
    #total transaction counts
    count = st.number_input('Total_Trans_Ct')
    #total count hange 4 to 1
    count_c41 = st.number_input('Total_Ct_Chng_Q4_Q1')
    #avg utilization ratio 
    ratio = st.number_input('Avg_Utilization_Ratio')
  
    query = np.array([card,month_on_book,relationship,limit,month_inactive, contacts, balance, amount_c41, amount,count, count_c41,ratio])
    query = query.reshape(1, 12)           
    
    # Use the model to make predictions on the DataFrame
    predictions = model.predict(query)

        # Display the predictions
    st.write(predictions)


# In[ ]:





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

