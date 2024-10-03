# PART 1: Import necessary libraries
# sml model
import streamlit as st
import pandas as pd
import joblib

# importing data 
import requests
import io

# display models
import matplotlib.pyplot as plt

@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Loan Prediction 💡")

    # LOADING BAR:
    progress_bar = st.progress(2, text="Setting urls...")
    
    # Defination of url-paths
    url1 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_0.csv'
    url2 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_1.csv'
    url3 = 'https://raw.githubusercontent.com/JAdamHub/M1-Exam-Submission/refs/heads/main/data/kiva_loans_part_2.csv'

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

# Description of the model
st.markdown("""
### Loan Amount Prediction:
#### Target: loan_amount (Amount of Loan)

#### Problem Statement:

The objective is to predict the loan amount that a borrower can receive based on various factors such as the borrower's country, sector, activity type, number of borrowers, funding duration, and repayment term. By understanding these relationships, the model can provide insights into how these factors influence the size of loans that borrowers are likely to be granted.

#### Type of Model:
The model uses an ensemble approach, combining both Random Forest and XGBoost models. This ensemble method takes advantage of the strengths of both algorithms, providing a robust and accurate prediction of loan amounts by reducing overfitting and improving model performance. The target variable, `loan_amount`, is continuous, making regression the appropriate method.

#### Objective:
This ensemble model aims to assist lending platforms and financial institutions in estimating the appropriate loan amount for borrowers under different conditions. The model's insights can help improve loan approval strategies, ensure fair and data-driven lending decisions, and effectively manage borrower expectations. By accurately predicting loan amounts, the model helps lenders optimize resources, minimize risk, and better serve their customers.
""")



@st.cache_resource  # Cache the model to improve performance by avoiding reloading on every run
def load_model_objects():
    # Load the pre-trained ensemble model from the specified file
    ensemble_model = joblib.load('ensemble_model.joblib')
    return ensemble_model  # Return the loaded model for use in the app

# Call the function to load the ensemble model
ensemble_model = load_model_objects()

# Create a mapping of unique activities for each sector
# This helps us understand what activities are associated with each sector
sector_activity_mapping = data.groupby('sector')['activity'].unique().to_dict()

# Create a mapping of unique regions for each country
# This provides insights into the regional distribution of countries in the dataset
country_region_mapping = data.groupby('country')['region'].unique().to_dict()

# Assuming the first estimator in the ensemble is the one with the preprocessor
# Extract the preprocessor from the first estimator of the ensemble model
preprocessor = ensemble_model.estimators_[0].named_steps['preprocessor']

# Extract the One-Hot Encoder and Scaler from the preprocessor
ohe = preprocessor.named_transformers_['cat']  # One-Hot Encoder for categorical features
scaler = preprocessor.named_transformers_['num']  # Scaler for numerical features

# Map feature names to their corresponding categories for easier reference
feature_names = ohe.feature_names_in_  # Get the feature names from the One-Hot Encoder
categories = ohe.categories_  # Get the categories for each feature
feature_categories = dict(zip(feature_names, categories))  # Create a dictionary mapping feature names to their categories

# Create two columns in the Streamlit layout for better organization
col1, col2 = st.columns(2)  

with col1:  # Using the first column of the layout
    # Sector selection dropdown
    sector = st.selectbox('Sector 🛄', options=feature_categories['sector'])
    
    # Get activities associated with the selected sector
    activities = sector_activity_mapping.get(sector, [])  # If the sector is not found, return an empty list
    # Activity selection dropdown based on the selected sector
    activity = st.selectbox('Activity 🚩', options=activities)
    
    # Country selection dropdown
    country = st.selectbox('Country 🌎', options=feature_categories['country'])
    
    # Get the region associated with the selected country
    country_region = country_region_mapping.get(country, [])  # If the country is not found, return an empty list
    # Region selection dropdown based on the selected country
    region = st.selectbox('Region 🛣️', options=country_region)
    
    # Gender selection radio buttons
    gender = st.radio('Gender 🧑‍🧒‍🧒', options=feature_categories['gender_class'])

with col2:  # Using the second column of the layout
    # Number input for the count of borrowers
    borrowers_count = st.number_input('Number of Borrowers 🧑‍🤝‍🧑', min_value=1, max_value=30, value=1)
    
    # Number input for the funding duration in days
    funding_duration = st.number_input('Funding Duration (Days) 🔔', min_value=0, max_value=90, value=2)
    
    # Number input for the loan term in months
    term_in_months = st.number_input('Term in Months 🧮', min_value=1, max_value=144, value=12)

# Prediction button
if st.button('Predict Loan Amount 🚀'):  # When the user clicks the button, the prediction process begins
    # Prepare all features into a DataFrame
    input_features = pd.DataFrame({
        'activity': [activity],  # User-selected activity
        'sector': [sector],      # User-selected sector
        'country': [country],    # User-selected country
        'region': [region],      # User-selected region
        'gender_class': [gender],  # User-selected gender
        'borrowers_count': [borrowers_count],  # User input for the number of borrowers
        'funding_duration_days': [funding_duration],  # User input for funding duration in days
        'term_in_months': [term_in_months],  # User input for loan term in months
    })

    # Make prediction directly using the ensemble model
    predicted_price = ensemble_model.predict(input_features)[0]  # Predict the loan amount based on input features

    # Display prediction in the Streamlit app
    st.metric(label="Predicted Loan Amount", value=f'${round(predicted_price, 2)}')  # Show the predicted loan amount rounded to two decimal places

    # Optional debugging statements
    print("Input features:", input_features.columns.tolist())  # Print the names of the input features for debugging
    print("Input features shape:", input_features.shape)  # Print the shape of the input features DataFrame for debugging

    # Load test data
    y_test = pd.read_csv("data/y_test.csv")
    X_test = pd.read_csv("data/X_test.csv")

    import matplotlib.pyplot as plt
    import seaborn as sns  # Seaborn for more styling options

    # Step 1: Make predictions on the test set using your Ensemble model
    y_pred = ensemble_model.predict(X_test)

    # Step 2: Plot actual vs predicted values with transparency and reduced marker size
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='blue', label='Predictions')  # Reduced marker size (s=10), transparency (alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Ideal Fit')

    # Add the predicted value from the user input as a green point
    plt.scatter([predicted_price], [predicted_price], color='green', s=100, label='Your Prediction')

    plt.title('Ensemble Model: Actual vs Predicted Loan Amounts', fontsize=16)
    plt.xlabel('Actual Loan Amount', fontsize=14)
    plt.ylabel('Predicted Loan Amount', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Use st.pyplot to render the plot in Streamlit
    st.pyplot(plt)
    
    st.markdown('Ensemble Model Training 🏋🏽‍♀️ R2 Score: 0.7806')
    st.markdown('Ensemble Model Test 👩🏽‍⚕️ R2 Score: 0.6822')