# PART 1: Import necessary libraries
# sml model
import streamlit as st
import pandas as pd
import joblib

# importing data libs
import requests
import io
import os

# display models
import matplotlib.pyplot as plt

# App Name & Icon (current page)
st.set_page_config(page_title="Loan Prediction", page_icon=":rocket:")

@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Loan Prediction 💡")

    # LOADING BAR:
    progress_bar = st.progress(2, text="Progress...")
    
    # define file paths for local files
    file_paths = [
        "data/kiva_loans_part_0.csv",
        "data/kiva_loans_part_1.csv",
        "data/kiva_loans_part_2.csv"
    ]

    # check if all files exist if yes we load them!! using if and else statement
    if all(os.path.exists(file) for file in file_paths):
        # if all files are downloaded, load them
        progress_bar.progress(83, text="Importing partial datasets...")
        data_part1 = pd.read_csv(file_paths[0])
        progress_bar.progress(85, text="Importing partial datasets...")
        data_part2 = pd.read_csv(file_paths[1])
        progress_bar.progress(87, text="Importing partial datasets...")
        data_part3 = pd.read_csv(file_paths[2])
        
    else:
        # if files are not downloaded, download them! 
    
        # define url-paths
        url1 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_0.csv'
        url2 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_1.csv'
        url3 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_2.csv'

        # loading the urls into requests to download data
        progress_bar.progress(40, text="Downloading datasets...1/3")
        response1 = requests.get(url1)
        progress_bar.progress(55, text="Downloading datasets...2/3")
        response2 = requests.get(url2)
        progress_bar.progress(75, text="Downloading datasets...3/3")
        response3 = requests.get(url3)

        # loading partial datasets
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

# Description of the model
st.markdown("""
### Loan Amount Prediction:
#### Target: loan_amount (Amount of Loan)

#### Problem Statement:

The objective is to predict the loan amount that a borrower can receive based on various factors such as the borrower's country, sector, activity type, number of borrowers, funding duration, and repayment term. By understanding these relationships, the model can provide insights into how these factors influence the size of loans that borrowers are likely to be granted.

#### Type of Model:
The model uses an ensemble approach, combining both Random Forest (30%) and XGBoost (70%) model. This ensemble method takes advantage of the strengths of both algorithms, providing a robust and accurate prediction of loan amounts by reducing overfitting and improving model performance. The target variable, `loan_amount`, is continuous, making regression the appropriate method.

#### Objective:
This ensemble model aims to assist lending platforms and financial institutions in estimating the appropriate loan amount for borrowers under different conditions. The model's insights can help improve loan approval strategies, ensure fair and data-driven lending decisions, and effectively manage borrower expectations. By accurately predicting loan amounts, the model helps lenders optimize resources, minimize risk, and better serve their customers.
""")



@st.cache_resource  # cache the model to improve performance and avoid reloading on every run
def load_model_objects():
    # load the pre-trained ensemble model from the specified file
    ensemble_model = joblib.load('ensemble_model.joblib')  # exported from M1-Exam_Prediction_Model.ipynb
    return ensemble_model  # return the loaded model for use in the app

# call the function to load the ensemble model
ensemble_model = load_model_objects()

# create a mapping of unique activities for each sector
# helps us understand which activities are associated with each sector
sector_activity_mapping = data.groupby('sector')['activity'].unique().to_dict()

# create a mapping of unique regions for each country
# gives insights into the regional distribution of countries in the dataset
country_region_mapping = data.groupby('country')['region'].unique().to_dict()

# assuming the first estimator in the ensemble is the one with the preprocessor
# extract the preprocessor from the first estimator of the ensemble model
preprocessor = ensemble_model.estimators_[0].named_steps['preprocessor']

# extract the one-hot encoder and scaler from the preprocessor
ohe = preprocessor.named_transformers_['cat']  # one-hot encoder for categorical features
scaler = preprocessor.named_transformers_['num']  # scaler for numerical features

# map feature names to their corresponding categories for easier reference
feature_names = ohe.feature_names_in_  # get the feature names from the one-hot encoder
categories = ohe.categories_  # get the categories for each feature
feature_categories = dict(zip(feature_names, categories))  # create a dictionary mapping feature names to their categories

# create two columns in the streamlit layout for better organization
col1, col2 = st.columns(2)  

with col1:  # using the first column of the layout
    # sector selection dropdown
    sector = st.selectbox('Sector 🛄', options=feature_categories['sector'])
    
    # get activities associated with the selected sector
    activities = sector_activity_mapping.get(sector, [])  # if the sector is not found, return an empty list
    # activity selection dropdown based on the selected sector
    activity = st.selectbox('Activity 🚩', options=activities)
    
    # country selection dropdown
    country = st.selectbox('Country 🌎', options=feature_categories['country'])
    
    # get the region associated with the selected country
    country_region = country_region_mapping.get(country, [])  # if the country is not found, return an empty list
    # region selection dropdown based on the selected country
    region = st.selectbox('Region 🛣️', options=country_region)
    
    # gender selection radio buttons
    gender = st.radio('Gender 🧑‍🧒‍🧒', options=feature_categories['gender_class'])

with col2:  # using the second column of the layout
    # number input for the count of borrowers
    borrowers_count = st.number_input('Number of Borrowers 🧑‍🤝‍🧑', min_value=1, max_value=30, value=1)
    
    # number input for the funding duration in days
    funding_duration = st.number_input('Funding Duration (Days) 🔔', min_value=0, max_value=90, value=2)
    
    # number input for the loan term in months
    term_in_months = st.number_input('Term in Months 🧮', min_value=1, max_value=144, value=12)

# prediction button
if st.button('Predict Loan Amount 🚀'):  # when the user clicks the button, the prediction process begins
    # prepare all INPUT USER SELECTED features into a dataframe
    input_features = pd.DataFrame({
        'activity': [activity],  
        'sector': [sector],      
        'country': [country],    
        'region': [region],      
        'gender_class': [gender],  
        'borrowers_count': [borrowers_count], 
        'funding_duration_days': [funding_duration], 
        'term_in_months': [term_in_months], 
    }) ## ^^^^ input_features dataframe is created from the user's input selections

    # make prediction directly using the ensemble model
    predicted_price = ensemble_model.predict(input_features)[0]  # predict the loan amount based on input features

    # display prediction in the streamlit app
    st.metric(label="Predicted Loan Amount", value=f'${round(predicted_price, 2)}')  # show the predicted loan amount rounded to two decimal places

    # optional debugging statements
    print("Input features:", input_features.columns.tolist())  # print the names of the input features for debugging
    print("Input features shape:", input_features.shape)  # print the shape of the input features dataframe for debugging

    # load test data
    y_test = pd.read_csv("data/y_test.csv")
    X_test = pd.read_csv("data/X_test.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns  # seaborn for more styling options

    # make predictions on the test set using the ensemble model
    y_pred = ensemble_model.predict(X_test)

    # plot actual vs predicted values with transparency and reduced marker size
    plt.figure(figsize=(14, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='blue', label='Real data points')  # reduced marker size (s=10), transparency (alpha=0.3)
    
    # add a red dashed line to represent the ideal fit line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', alpha=0.5, label='Ideal Fit')
    
    # add the predicted value from the user input as a green point
    plt.scatter([predicted_price], [predicted_price], color='green', s=100, label='Your Prediction')

    plt.title('Ensemble Model: Actual vs Predicted Loan Amounts', fontsize=16)
    plt.xlabel('Actual Loan Amount', fontsize=14)
    plt.ylabel('Predicted Loan Amount', fontsize=14)
    plt.legend()
    plt.grid(True)

    # use st.pyplot to render the plot in streamlit
    st.pyplot(plt)
    
st.markdown('Ensemble Model Training 🏋🏽‍♀️ R2 Score: 0.7806')
st.markdown('Ensemble Model Test 👩🏽‍⚕️ R2 Score: 0.6822')
