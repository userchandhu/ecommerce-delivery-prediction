import streamlit as st
from joblib import load
import pandas as pd

model = load('gb_model_file.joblib')
scaler = load('scaler_file.joblib')

st.title('E-Commerce Product Delivery Detection by Prem')

st.header('Enter order Details')

warehouse_block = st.selectbox('Warehouse block',['A','B','C','D','F'])
Mode_of_Shipment = st.selectbox('Mode of Shipment',['Flight','Ship','Road'])
Customer_care_calls = st.number_input('Customer care calls',min_value=0)
Customer_rating = st.number_input('Customer rating',min_value=0, max_value=5)
Cost_of_the_Product = st.number_input('Cost of the Product')
Prior_purchases = st.number_input('Prior purchases', min_value=0)
Product_importance = st.selectbox('Product importance',['low','medium','high'])
Gender = st.selectbox('Gender',['F','M'])
Discount_offered = st.number_input('Discount offered')
Weight_in_gms = st.number_input('Weight in gms')


# - FEATURE ENGINEERING -
# create new features same as training

Cost_per_gram = Cost_of_the_Product / Weight_in_gms if Weight_in_gms != 0 else 0
Discount_percent = Cost_of_the_Product / Discount_offered if Discount_offered != 0 else 0
Call_intensity = Customer_care_calls / (Prior_purchases + 1)

# label encoding
Product_importance = {'low':0, 'medium':1, 'high':2}[Product_importance]
Gender = {'F':0, 'M':1}[Gender]

#One hot encoding (drop_first=True was used in training)
#so warehouse block (A is dropped)

wb_B = 1 if warehouse_block == 'B' else 0
wb_C = 1 if warehouse_block == 'C' else 0
wb_D = 1 if warehouse_block == 'D' else 0
wb_F = 1 if warehouse_block == 'F' else 0

# shipment (Flight is dropped)
ms_Road = 1 if Mode_of_Shipment == 'Road' else 0
ms_Ship = 1 if Mode_of_Shipment == 'Ship' else 0


# - CREATE INPUT DATA -
# create dataframe in same format as training

data =data = pd.DataFrame([[ 
    Customer_care_calls,
    Customer_rating,
    Cost_of_the_Product,
    Prior_purchases,
    Product_importance,
    Gender,
    Discount_offered,
    Weight_in_gms,
    Cost_per_gram,
    Discount_percent,
    Call_intensity,
    wb_B, wb_C, wb_D, wb_F,
    ms_Road, ms_Ship
]], columns=[
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Product_importance',
    'Gender',
    'Discount_offered',
    'Weight_in_gms',
    'Cost_per_gram',
    'Discount_percent',
    'Call_intensity',
    'Warehouse_block_B',
    'Warehouse_block_C',
    'Warehouse_block_D',
    'Warehouse_block_F',
    'Mode_of_Shipment_Road',
    'Mode_of_Shipment_Ship'
])

# - SCALING -
# numeric columns used in training

numeric_columns = [
 'Customer_care_calls',
 'Customer_rating',
 'Cost_of_the_Product',
 'Prior_purchases',
 'Discount_offered',
 'Weight_in_gms',
 'Cost_per_gram',
 'Discount_percent',
 'Call_intensity'
]

data[numeric_columns] = scaler.transform(data[numeric_columns])

# - PREDICTION -
if st.button('Predict'):
    prediction = model.predict(data)
    if prediction[0]==1 :
        st.error('Product NOT Delivered On Time')
    else:
        st.success('Product Delivered On Time')


