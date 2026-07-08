import pandas as pd
import numpy as np 
import streamlit as st
from sklearn.preprocessing import LabelEncoder , OneHotEncoder ,StandardScaler
import pickle 
import tensorflow as tf

model=tf.keras.models.load_model('model.h6')



with open('geo_encode.pkl' ,'rb') as file:
   geo_encode = pickle.load(file)
   
with open('scalar.pkl' ,'rb') as file:
   scalar = pickle.load(file)


with open('label_encode_gender.pkl' ,'rb') as file:
   label_encode_gender=pickle.load(file)


## streamlit model

st.title('customer_salary_prediction')

# User input
geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)



gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
stay= st.selectbox('exitin', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [stay]
    
})


geo_encoded = geo_encode.transform([[geography]]).toarray()
geo_encode_df= pd.DataFrame(geo_encoded,columns=geo_encode.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data,geo_encode_df],axis=1)
input_scaled=scalar.transform(input_data)


prediction=model.predict(input_data)


st.write(f"salary prediction is:{prediction.item()}")

prediction_proba =prediction[0][0]


   