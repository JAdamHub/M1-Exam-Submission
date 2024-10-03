# PART 1: Import necessary libraries
# sml model
import streamlit as st
import pandas as pd
import joblib

# importing data 
import requests
import io

@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Microloans Statistics 🪙")

    # LOADING BAR:
    progress_bar = st.progress(2, text="Setting urls...")
    
    # Defination of url-paths
    url1 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/kiva_loans_part_0.csv'
    url2 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/kiva_loans_part_1.csv'
    url3 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/kiva_loans_part_2.csv'

    # Loading the urls into requests to download data
    progress_bar.progress(40, text="Downloading datasets...1/3")
    response1 = requests.get(url1)
    progress_bar.progress(55, text="Downloading datasets...2/3")
    response2 = requests.get(url2)
    progress_bar.progress(75, text="Downloading datasets...3/3")
    response3 = requests.get(url3)

    # Loading partial datasets
    progress_bar.progress(83, text="Importing partial datasets...")
    data_part1 = pd.read_csv(io.StringIO(response1.text))
    progress_bar.progress(85, text="Importing partial datasets...")
    data_part2 = pd.read_csv(io.StringIO(response2.text))
    progress_bar.progress(87, text="Importing partial datasets...")
    data_part3 = pd.read_csv(io.StringIO(response3.text))

    # Combining the datasets into one df using pd.concat
    progress_bar.progress(89, text="Merging datasets...")
    data = pd.concat([data_part1, data_part2, data_part3])
    
# Drop columns we're not going to use
    progress_bar.progress(91, text="Dropping irrelevant columns & cleaning dataset...")
    data = data.drop(['tags', 'use', 'currency', 'country_code', 'partner_id'], axis=1)

#Dropping missing values using dropna
    data.dropna(inplace=True)
    

# Selecting top 20 countries
    # what is the total amount of loan_amount for each country?
    country_loans = data.groupby('country')['loan_amount'].sum()

    # we would like the top 20 countries
    country_top_20 = country_loans.sort_values(ascending=False).head(20)

    # using index we create a new variable with the top 20 countries
    data = data[data['country'].isin(country_top_20.index)]
    progress_bar.progress(100, None) 
    
    return data

data = loading_dataset()

import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model_objects():
    ensemble_model = joblib.load('ensemble_model.joblib')
    return ensemble_model

ensemble_model = load_model_objects()

# 
sector_activity_mapping = data.groupby('sector')['activity'].unique().to_dict() 
country_region_mapping = data.groupby('country')['region'].unique().to_dict()

# Assuming the first estimator in the ensemble is the one with the preprocessor
preprocessor = ensemble_model.estimators_[0].named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat']
scaler = preprocessor.named_transformers_['num']

# Map feature names to categories
feature_names = ohe.feature_names_in_
categories = ohe.categories_
feature_categories = dict(zip(feature_names, categories))

col1, col2 = st.columns(2)

with col1:
    # Sector
    sector = st.selectbox('Sector', options=feature_categories['sector'])
    
    # Get activities for the selected sector
    activities = sector_activity_mapping.get(sector, [])
    activity = st.selectbox('Activity', options=activities)
    
    # Country
    country = st.selectbox('Country', options=feature_categories['country'])
    
    # Get region for the selected country
    country_region = country_region_mapping.get(country, [])
    region = st.selectbox('Region', options=country_region)
    
    # Gender
    gender = st.radio('Gender', options=feature_categories['gender_class'])

with col2:
    borrowers_count = st.number_input('Number of Borrowers', min_value=1, max_value=30, value=1)
    funding_duration = st.number_input('Funding Duration (Days)', min_value=0, max_value=90, value=30)
    term_in_months = st.number_input('Term in Months', min_value=1, max_value=144, value=12)

# Prediction button
if st.button('Predict Loan Amount 🚀'):
    # Prepare all features
    input_features = pd.DataFrame({
        'activity': [activity],
        'sector': [sector],
        'country': [country],
        'region': [region],
        'gender_class': [gender],
        'borrowers_count': [borrowers_count],
        'funding_duration_days': [funding_duration],
        'term_in_months': [term_in_months],
    })

    # Make prediction directly using the ensemble model
    predicted_price = ensemble_model.predict(input_features)[0]

    # Display prediction
    st.metric(label="Predicted Loan Amount", value=f'${round(predicted_price, 2)}')

    # Optional debugging statements
    print("Input features:", input_features.columns.tolist())
    print("Input features shape:", input_features.shape)