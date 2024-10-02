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

# PART 1: Function to load the data parts and use pd.concat to combine the 3 parts to one dataset
@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Microloans Statistics 🪙")

    # LOADING BAR:
    progress_bar = st.progress(2, text="Setting urls...")
    
    # Defination of url-paths
    url1 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_0.csv.zip'
    url2 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_1.csv.zip'
    url3 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_2.csv.zip'

    # Loading the urls into requests to download data
    progress_bar.progress(9, text="Downloading datasets...1/3")
    response1 = requests.get(url1)
    progress_bar.progress(32, text="Downloading datasets...2/3")
    response2 = requests.get(url2)
    progress_bar.progress(50, text="Downloading datasets...3/3")
    response3 = requests.get(url3)

    # Saves the .zip data as files
    progress_bar.progress(55, text="Saving dataset zip-file...1/3")
    with open("kiva_loans_part_0.csv.zip", "wb") as file:
        file.write(response1.content)
    progress_bar.progress(60, text="Saving dataset zip-file...2/3")
    with open("kiva_loans_part_1.csv.zip", "wb") as file:
        file.write(response2.content)
    progress_bar.progress(65, text="Saving dataset zip-file...3/3")
    with open("kiva_loans_part_2.csv.zip", "wb") as file:
        file.write(response3.content)

    # Unzip the files to get .csv
    progress_bar.progress(70, text="Unzipping dataset...1/3")
    with zipfile.ZipFile("kiva_loans_part_0.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()
    progress_bar.progress(75, text="Unzipping dataset...2/3")
    with zipfile.ZipFile("kiva_loans_part_1.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()
    progress_bar.progress(81, text="Unzipping dataset...3/3")
    with zipfile.ZipFile("kiva_loans_part_2.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()

    # Loading partial datasets
    progress_bar.progress(83, text="Importing partial datasets...")
    data_part1 = pd.read_csv("kiva_loans_part_0.csv")
    progress_bar.progress(85, text="Importing partial datasets...")
    data_part2 = pd.read_csv("kiva_loans_part_1.csv")
    progress_bar.progress(87, text="Importing partial datasets...")
    data_part3 = pd.read_csv("kiva_loans_part_2.csv")

    # Combining the datasets into one df using pd.concat
    progress_bar.progress(89, text="Merging datasets...")
    data = pd.concat([data_part1, data_part2, data_part3])
    
    # PART 2: CLEANING DATA & MANIPULATION

    # Drop columns we're not going to use
    progress_bar.progress(91, text="Dropping irrelevant columns & cleaning dataset...")
    data = data.drop(['tags', 'use', 'currency', 'country_code', 'partner_id'], axis=1)

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
    ai_overview = DDGS().chat("You're a smart data analyst. Provide and interpretate an overview of the Dataset." + str((data.describe())))
    return ai_overview

ai_overview = explaination()
st.write(ai_overview)