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

# PART 3: Setting up title and filter-sideheader
st.sidebar.header("Filters 📊")
#########################################################################################################################
# GENDER SIDEBAR

# CREATE LIST OVER GENDERS
all_gender = data['gender_class'].unique().tolist() # - REMOVED DUE TO GENDER WONT BE UPDATED IN DATASET

# GENDER SIDEBAR MULTISELECT
selected_gender = st.sidebar.multiselect("Select Gender Group 🧑‍🧑‍🧒", all_gender, default=all_gender)

# Filtration of data based on sidebar
filtered_data = data[data['gender_class'].isin(selected_gender)]
#########################################################################################################################
# SECTOR SIDEBAR

# CREATE LIST OVER ALL SECTORS
all_sectors = data['sector'].unique().tolist()

# SECTOR SIDEBAR MULTISELECT
selected_sector = st.sidebar.multiselect("Select Sectors 💼", all_sectors, default=all_sectors)

# Filtration of data based on sidebar
filtered_data = filtered_data[filtered_data['sector'].isin(selected_sector)]
#########################################################################################################################
# COUNTRY SIDEBAR

# CALCULATE TOP 20 COUNTRIES & CREATE A LIST
top_countries = data.groupby('country').size().nlargest(20).index.tolist()

# COUNTRY SIDEBAR MULTISELECT
selected_country = st.sidebar.multiselect(
    "Select Country 🇺🇳", top_countries, default=top_countries)

# Filtration of data based on sidebar
filtered_data = filtered_data[filtered_data['country'].isin(selected_country)]
#########################################################################################################################
# CHECK IF CHOICE HAS BEEN MADE ON GENDER GROUP, SECTORS & COUNTRY
# GENDER - NO CHOICE WARNING 
if not selected_gender:
    st.warning("Please select a gender group from the sidebar ⚠️")
    st.stop()

# SECTOR - NO CHOICE WARNING 
if not selected_sector:
    st.warning("Please select a sector from the sidebar ⚠️")
    st.stop()

# COUNTRY - NO CHOICE WARNING 
if not selected_country:
    st.warning("Please select a country from the sidebar ⚠️")
    st.stop()

# Sidebar slider for selecting loan amount range
min_loan_amount = int(data['loan_amount'].min())
max_loan_amount = int(data['loan_amount'].max())

selected_amount_range = st.sidebar.slider(
    'Select Loan Amount Range',
    min_value=min_loan_amount,
    max_value=max_loan_amount,
    value=(min_loan_amount, max_loan_amount),
    step=1
)

# Filter the dataframe based on the selected loan amount range
filtered_data = data[
    (data['loan_amount'] >= selected_amount_range[0]) &
    (data['loan_amount'] <= selected_amount_range[1])
]


#########################################################################################################################
# PART 4: VISUALIZATIONS
# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Records of Loans Issued By Sector & Country (Top 20 Countries)", 
     "KDE Plot - By Sector, Country & Total",
     "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country", 
     "Heatmap of Average Loan by Sector & Country",
     "Frequency of Funded Loans Over Time"])

if visualization_option == "Records of Loans Issued By Sector & Country (Top 20 Countries)":
    # Bar chart for Records of Loans Issued By Sector (Top 20 Countries)
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[0, filtered_data['loan_amount'].max()])),  # Set x-axis to start at 0
        y='count()',
        color='sector',
    ).properties(
        title='Records of Loans Issued By Sector & Country (Top 20 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)


 # Bar chart for Records of Loans Issued By Country (Top 20 Countries)
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[0, filtered_data['loan_amount'].max()])),  # Force x-axis to start at 0
        y='count()',
        color='country',
    ).properties(
        title='Records of Loans Issued By Country Only (Top 20 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)



elif visualization_option == "KDE Plot - By Sector, Country & Total":
# KDE plot - SECTOR
    plt.figure(figsize=(17, 7))
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='sector', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Loan Amount by Sector')
    plt.xlim(left=0)  # Ensure the x-axis starts at 0
    st.pyplot(plt)

# KDE plot - Country
    plt.figure(figsize=(17, 7))
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='country', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Loan Amount by Country')
    plt.xlim(left=0)  # Ensure the x-axis starts at 0
    st.pyplot(plt)

# KDE plot - TOTAL
    plt.figure(figsize=(17, 7))
    sns.kdeplot(data=filtered_data, x='loan_amount', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Total Loan Amount')
    plt.xlim(left=0)  # Ensure the x-axis starts at 0
    st.pyplot(plt)


elif visualization_option == "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country":
    gender_sector = filtered_data.groupby(['sector', 'gender_class'])['loan_amount'].mean().unstack()
    gender_sector.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))
    plt.title('GENDER & SECTOR ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Sector')
    plt.ylabel('Sector')
    plt.xlabel('Mean Loan Amount')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    gender_country = filtered_data.groupby(['country', 'gender_class'])['loan_amount'].mean().unstack()
    gender_country.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))
    plt.title('GENDER & COUNTRY ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Country')
    plt.ylabel('Country')
    plt.xlabel('Mean Loan Amount')
    plt.xticks(rotation=0)
    st.pyplot(plt)

elif visualization_option == "Heatmap of Average Loan by Sector & Country":
    heatmap_data = filtered_data.pivot_table(index='sector', columns='country', values='loan_amount', aggfunc='mean')
    plt.figure(figsize=(20, 12))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")
    plt.title('Heatmap of Average Loan by Sector & Country')
    st.pyplot(plt)

elif visualization_option == "Frequency of Funded Loans Over Time":
    time_data = filtered_data
    
    # CONVERTING 'funded_time' column to datetime format
    time_data['funded_time'] = pd.to_datetime(time_data['funded_time'])
    time_data.set_index('funded_time', inplace=True)

    # Resample the data to a monthly frequency (can also be yearly, daily, etc.)
    funded_trend = time_data.resample('M').size()

    # Create the plot
    plt.figure(figsize=(20, 6))
    plt.plot(funded_trend, label='Total Funded Loans', color='blue')

    # Add labels, title, and legend
    plt.title('Frequency of Funded Loans Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Funded Loans')
    plt.legend()
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(plt)


# PART BONUS - DEBUGGING
with st.expander("DEBUGGING 🤓"):
    st.write("Selected sectors:", selected_sector)
    st.write("Selected countries:", selected_country)
    st.write("Filtered data:", filtered_data)