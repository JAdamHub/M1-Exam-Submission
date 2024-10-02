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

# SML libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    data = data.drop(['tags', 'use', 'currency', 'country_code'], axis=1)

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

# CALCULATE TOP 10 COUNTRIES & CREATE A LIST
top_countries = data.groupby('country').size().nlargest(10).index.tolist()

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
#########################################################################################################################
# PART 4: DATA OVERVIEW
with st.expander("GENERAL OVERVIEW OF DATA & DESCRIPTIVE STATISTICS (all data 🗺️)"):
    st.header("Dataset Overview (all data)")
    st.markdown("data.head():")
    st.table(data.head())
    st.header("Descriptive Statistics (all data)")
    st.markdown("data.describe().T")
    st.dataframe(data.describe().T)

# PART 4.5: DESCRIPTIVE STATISTICS 
with st.expander("FILTERED DESCRIPTIVE STATISTICS (side-filtered data 📊)"):
    st.header("Descriptive Statistics (based on sidebar-filter)")
    st.markdown("filtered_data.describe().T")
    st.dataframe(filtered_data.describe().T)

#########################################################################################################################
# PART 5: VISUALIZATIONS
# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Records of Loans Issued By Sector & Country (Top 10 Countries)", 
     "KDE Plot - By Sector, Country & Total",
     "Box Plot - Country, Sector & Gender Group",
     # "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country", REMOVED
     "Heatmap of Average Loan by Sector & Country",
     "Frequency of Funded Loans Over Time"])

if visualization_option == "Records of Loans Issued By Sector & Country (Top 10 Countries)":
    # Bar chart for Records of Loans Issued By Sector & Country (Top 10 Countries)
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x='loan_amount',
        y='count()',
        color='sector',
    ).properties(
        title='Records of Loans Issued By Sector & Country (Top 10 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)

    # Bar chart for Countries only
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x='loan_amount',
        y='count()',
        color='country',
    ).properties(
        title='Records of Loans Issued By Country Only (Top 10 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)

elif visualization_option == "KDE Plot - By Sector, Country & Total":
    # KDE plot - SECTOR
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='sector', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Loan Amount by Sector')
    st.pyplot(plt)

    # KDE plot - Country
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='country', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Loan Amount by Country')
    st.pyplot(plt)

    # KDE plot - TOTAL
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=filtered_data, x='loan_amount', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Total Loan Amount')
    st.pyplot(plt)

elif visualization_option == "Box Plot - Country, Sector & Gender Group":
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_data, x='sector', y='loan_amount', hue='gender_class', palette='gist_rainbow')
    plt.title('Box Plot of Loan Amounts By Sector and Gender Group')
    plt.xlabel('Sector')
    plt.ylabel('Loan Amount')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_data, x='sector', y='loan_amount', hue='country', palette='gist_rainbow')
    plt.title('Box Plot of Loan Amounts By Country (top 10) & Sector')
    plt.xlabel('Sector')
    plt.ylabel('Country')
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif visualization_option == "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country":
    gender_sector_country = filtered_data.groupby(['sector', 'country', 'gender_class'])['loan_amount'].mean().unstack()
    gender_sector_country.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 26))
    plt.title('Stacked Bar Chart of Mean Loan Amount by Gender Sector & Country')
    plt.ylabel('Sector and Country')
    plt.xlabel('Mean Loan Amount')
    plt.xticks(rotation=0)
    st.pyplot(plt)

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
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")
    plt.title('Heatmap of Average Loan by Sector & Country')
    st.pyplot(plt)

elif visualization_option == "Frequency of Funded Loans Over Time":
    time_data = filtered_data
    
    # CONVERTING 
    time_data['funded_time'] = pd.to_datetime(time_data['funded_time'])

    # Set date column as index:
    time_data.set_index('funded_time', inplace=True)

    #Resample the data to a monthly frequency (can also be yearly, daily, etc.)
    funded_trend = time_data.resample('M').size()

    #plot the frequency of searches over time
    plt.figure(figsize=(12, 6))
    plt.plot(funded_trend, label='Total Funded Loans', color='blue')

    #add labels and title
    plt.title('Frequency of Funded Loans Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Funded Loans')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# PART 6: Clustering Analysis
st.header("Clustering Analysis 🧮")

def clustering_and_visualization(data):
    st.write("This section applies unsupervised learning techniques such as K-Means and Hierarchical Clustering on the KIVA dataset.")

    # Preprocessing
    numerical_cols = ['loan_amount', 'lender_count']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_cols])

    # K-Means clustering
    n_clusters = st.slider("Select number of clusters for K-Means", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    # Visualize K-Means results
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans_labels, cmap='viridis')
    ax.set_title('K-Means Clustering Results')
    ax.set_xlabel('Standardized Loan Amount')
    ax.set_ylabel('Standardized Lender Count')
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Hierarchical Clustering
    st.subheader("Hierarchical Clustering")
    n_clusters_hc = st.slider("Select number of clusters for Hierarchical Clustering", min_value=2, max_value=10, value=3)
    hc = AgglomerativeClustering(n_clusters=n_clusters_hc)
    hc_labels = hc.fit_predict(scaled_data)

    # Visualize Hierarchical Clustering results
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=hc_labels, cmap='viridis')
    ax.set_title('Hierarchical Clustering Results')
    ax.set_xlabel('Standardized Loan Amount')
    ax.set_ylabel('Standardized Lender Count')
    plt.colorbar(scatter)
    st.pyplot(fig)

if st.button("Perform Clustering Analysis"):
    clustering_and_visualization(filtered_data)


if st.button("AI Assitant, explain!!!"):
    results = DDGS().chat("You're a smart data analyst. Provide and interpretate the results. Remember to check which Sectors, Gender groups, Country." + str((filtered_data.describe())))
    st.write(results)

# PART BONUS - DEBUGGING
with st.expander("DEBUGGING 🤓"):
    st.write("Selected sectors:", selected_sector)
    st.write("Selected countries:", selected_country)
    st.write("Filtered data:", filtered_data)