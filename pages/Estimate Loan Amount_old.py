# PART 0.5: Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import geopandas as gpd
import altair as alt
from vega_datasets import data
from duckduckgo_search import DDGS
import zipfile
import requests
import re
import io

### SML MODEL LIBS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Import the confusion matrix plotter module
from mlxtend.plotting import plot_confusion_matrix

#  Model selection & Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from imblearn.under_sampling import RandomUnderSampler

# pipeline for the different models
from sklearn.pipeline import Pipeline

# decision Tree
from sklearn.tree import DecisionTreeRegressor

# tabular data explanation with LIME
import lime.lime_tabular  

# PART 1: Function to load the data parts and use pd.concat to combine the 3 parts to one dataset
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
    
    # PART 2: CLEANING DATA & MANIPULATION

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
    
#Removal of outliers
    progress_bar.progress(93, text="Removing outliers...")
    z_scores = zscore(data['loan_amount'])

    # Get boolean array indicating the presence of outliers
    # Using 2 & -2 z_scores to get 95% of data within 2 standard deviations
    data['outlier_loan_amount'] = (z_scores > 2) | (z_scores < -2)

    #Removing outliers
    data = data[~data['outlier_loan_amount']]

# GENDER CLASSIFICATION (GROUPING GENDERS)
    progress_bar.progress(93, text="Creating gender groups...")
    loan_gender = data

    # COUNT MALE & FEMALE BORROWERS 
    progress_bar.progress(94, text="Creating gender groups...")
    loan_gender['male_borrowers'] = loan_gender['borrower_genders'].apply(lambda x: len(re.findall(r'\bmale', x)))
    loan_gender['female_borrowers'] = loan_gender['borrower_genders'].apply(lambda x: len(re.findall(r'\bfemale', x)))

    # CALCULATE TOTAL BORROWER COUNT
    progress_bar.progress(96, text="Calculating gender groups...")
    loan_gender['borrowers_count'] = loan_gender['male_borrowers'] + loan_gender['female_borrowers']

    # HANDLE SITUATIONS WHERE 'BORROWERS_COUNT' IS 0 TO AVOID DIVISION BY 0
    progress_bar.progress(97, text="Creating gender groups...")
    loan_gender['male_borrower_ratio'] = loan_gender['male_borrowers'] / loan_gender['borrowers_count'].replace(0, 1)

    # FUNCTION TO CLASSIFY GENDER BASED ON RATIO
    def classify_genders(ratio):
        if ratio == 1:
            return 'male group'
        elif ratio == 0:
            return 'female group'
        else:
            return 'mixed group'
    progress_bar.progress(98, text="Applying gender mapping...")
    # APPLY GENDER CLASSIFICATION
    data['gender_class'] = loan_gender['male_borrower_ratio'].apply(classify_genders)

### FUNDING DURATION ###
    # We would like to also include the time/duration between posted_time and funded_time - in other words: how long it takes to get a loan funded
    # convert to pd.datetime
    data['posted_time'] = pd.to_datetime(data['posted_time'])
    data['funded_time'] = pd.to_datetime(data['funded_time'])

    # calculate time between posted_time and funded_time
    data['funding_duration'] = data['funded_time'] - data['posted_time']

    # the result in days instead
    data['funding_duration_days'] = (data['funded_time'] - data['posted_time']).dt.total_seconds() / (24 * 60 * 60)
    
### Data imbalance fix ###

    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=9000)

    # Apply resampling using 'country' as a balancing feature
    X_resampled, _ = rus.fit_resample(data, data['country'])

    # updating dataframe
    data = X_resampled

#Done importing
### CREATING MODEL ###

    # define the target variable 'y' (loan_amount)
    y = data['loan_amount']

    # define the feature set 'X' (all other columns except loan_amount)
    X = data[['activity', 'sector', 'country', 'borrowers_count', 'funding_duration_days', 'region', 'gender_class', 'term_in_months']]

    # preprocessing for numeric features
    numeric_features = ['borrowers_count', 'funding_duration_days', 'term_in_months']
    numeric_transformer = StandardScaler()

    # preprocessing for categorical features
    categorical_features = ['activity', 'sector', 'country', 'region', 'gender_class']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    ### PIPELINE CREATION ###
    # Pipeline for Random Forest Regression
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, 
                                            min_samples_split=5, 
                                            min_samples_leaf=3,
                                            max_depth=20))
    ])

    # Pipeline for XGBoost
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())
    ])

    ### VOTING ENSAMBLE MODEL ###
    # create a votingregressor that combines the two pipeline models
    ensemble_model = VotingRegressor(
        estimators=[('rf', pipeline_rf), ('xgb', pipeline_xgb)],
        weights=[0.3, 0.7]  # 30% Random Forest & 70% XGBoost 
        # ^^^ possible to adjust weights based on model performance
    )

    # fit the ensemble model
    ensemble_model.fit(X_train, y_train)

    progress_bar.progress(100, None)

    return data, ensemble_model, X_test, y_test, y_train, X_train

data, ensemble_model, X_test, y_test, y_train, X_train = loading_dataset()

import streamlit as st
import pandas as pd
import pickle

# Load the loans DataFrame and pipeline_rf model
# Replace 'loans.csv' and 'pipeline_rf.pkl' with your actual file paths
loans = pd.read_csv('loans.csv')
with open('pipeline_rf.pkl', 'rb') as f:
    pipeline_rf = pickle.load(f)

# Create a dictionary mapping countries to their regions and sectors to activities
country_region_map = loans.groupby('country')['region'].unique().to_dict()
sector_activity_map = loans.groupby('sector')['activity'].unique().to_dict()

# Unique values for dropdowns
unique_gender = loans['gender_class'].unique().tolist()
unique_sector = loans['sector'].unique().tolist()
unique_country = loans['country'].unique().tolist()

# Define the prediction function
def predict_rf(gender_class, activity, sector, country, region, borrowers_count, funding_duration_days, term_in_months):
    # Create a dictionary of input values
    input_data = {
        'gender_class': [gender_class],
        'sector': [sector],
        'activity': [activity],
        'country': [country],
        'region': [region],
        'borrowers_count': [borrowers_count],
        'funding_duration_days': [funding_duration_days],
        'term_in_months': [term_in_months]
    }
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame(input_data)
    # Make a prediction
    prediction = pipeline_rf.predict(input_df)[0]
    # Return the prediction
    return prediction

# Streamlit app
st.title('Loan Amount Predictor')
st.write('Input the needed variables to get your estimated loan amount.')

# Gender Class
gender_class = st.selectbox('Gender Class', unique_gender)

# Sector
sector = st.selectbox('Sector', unique_sector)

# Activity - depends on sector
activities = sector_activity_map.get(sector, [])
activity = st.selectbox('Activity', activities)

# Country
country = st.selectbox('Country', unique_country)

# Region - depends on country
regions = country_region_map.get(country, [])
region = st.selectbox('Region', regions)

# Number of Borrowers
borrowers_count = st.slider('Number of Borrowers', min_value=1, max_value=10, value=1, step=1)

# Funding Duration in Days
funding_duration_days = st.slider('Funding Duration (Days)', min_value=0, max_value=90, value=30, step=1)

# Term in Months
term_in_months = st.slider('Term in Months', min_value=1, max_value=144, value=12, step=1)

# Submit button
if st.button('Submit'):
    # Make the prediction
    prediction = predict_rf(
        gender_class, activity, sector, country, region,
        borrowers_count, funding_duration_days, term_in_months
    )
    # Display the result
    st.success(f'Predicted Loan Amount: ${prediction:,.2f}')
