# PART 0.5: Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from duckduckgo_search import DDGS
import requests
import re
import io
import os

# App Name & Icon
st.set_page_config(page_title="Kiva Microloans", page_icon=":rocket:")

# PART 1: Function to load the data parts and use pd.concat to combine the 3 parts to one dataset
@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Microloans 🪙")

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
    
    # PART 2: CLEANING DATA & MANIPULATION

    # Drop columns we're not going to use
    progress_bar.progress(91, text="Dropping irrelevant columns & cleaning dataset...")
    data = data.drop(['tags', 'use', 'currency', 'country_code', 'partner_id', 'id', 'funded_amount'], axis=1)

    #Dropping missing values using dropna
    data.dropna(inplace=True)

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

    #Done
    progress_bar.progress(100, None)

    return data

data = loading_dataset()

#########################################################################################################################
# PART 4: DATA OVERVIEW

with st.expander("OVERVIEW OF DATA 🗺️"):
    st.subheader("Dataset Overview (cleaned data)")
    st.markdown("data.head():")
    st.table(data.head())
with st.expander("DESCRIPTIVE STATISTICS 📊"):
    st.subheader("Descriptive Statistics (cleaned data)")
    st.markdown("data.describe().T")
    st.dataframe(data.describe())

# AI explaination
@st.cache_data
def explaination():
    ai_overview = DDGS().chat("You're a smart data analyst. Provide and interpretate an overview of the KIVA Microloans Dataset. Make sure text is clear after numbers are shown" + str((data.describe())), model='claude-3-haiku')
    return ai_overview

ai_overview = explaination()
st.write(ai_overview)