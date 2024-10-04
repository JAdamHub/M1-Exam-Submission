# PART 0.5: Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import altair as alt

# data importing libs
import requests
import re
import io
import os

# App Name & Icon
st.set_page_config(page_title="Data Exploration", page_icon=":rocket:")

# PART 1: Function to load the data parts and use pd.concat to combine the 3 parts to one dataset
@st.cache_data  # Cache the function to enhance performance - tells streamlit to keep the dataset in memory/cache
def loading_dataset():
    # Setting Title
    st.title("💰 KIVA - Data Exploration 📊")
    
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

    # We're now importing the dataset parts into pandas DataFrames.
    
    # Combining the datasets into one df using pd.concat
    progress_bar.progress(89, text="Merging datasets...")
    data = pd.concat([data_part1, data_part2, data_part3])
    
    # This combines all three DataFrames into a single dataframe for analysis.

    # PART 2: CLEANING DATA & MANIPULATION

    # Drop columns we're not going to use
    progress_bar.progress(91, text="Dropping irrelevant columns & cleaning dataset...")
    data = data.drop(['tags', 'use', 'currency', 'country_code', 'partner_id'], axis=1)

    # We remove columns that won't be necessary for our analysis, like 'tags' and 'currency'.

    #Dropping missing values using dropna
    data.dropna(inplace=True)

    #Removal of outliers
    progress_bar.progress(93, text="Removing outliers...")
    z_scores = zscore(data['loan_amount'])

    # We calculate z-scores to detect and remove outliers in the 'loan_amount' column.
    # Z-scores >2 or <-2 indicate the presence of outliers.

    # Removing outliers
    data['outlier_loan_amount'] = (z_scores > 2) | (z_scores < -2)
    data = data[~data['outlier_loan_amount']]
    
    # Now we're filtering out those outliers from the data.

# GENDER CLASSIFICATION (GROUPING GENDERS)
    progress_bar.progress(93, text="Creating gender groups...")
    loan_gender = data

    # COUNT MALE & FEMALE BORROWERS 
    progress_bar.progress(94, text="Creating gender groups...")
    loan_gender['male_borrowers'] = loan_gender['borrower_genders'].apply(lambda x: len(re.findall(r'\bmale', x)))
    loan_gender['female_borrowers'] = loan_gender['borrower_genders'].apply(lambda x: len(re.findall(r'\bfemale', x)))

    # Using regex, we count the male and female borrowers in the 'borrower_genders' column.

    # CALCULATE TOTAL BORROWER COUNT
    progress_bar.progress(96, text="Calculating gender groups...")
    loan_gender['borrowers_count'] = loan_gender['male_borrowers'] + loan_gender['female_borrowers']

    # This calculates the total number of borrowers by adding up males and females.

    # HANDLE SITUATIONS WHERE 'BORROWERS_COUNT' IS 0 TO AVOID DIVISION BY 0
    progress_bar.progress(97, text="Creating gender groups...")
    loan_gender['male_borrower_ratio'] = loan_gender['male_borrowers'] / loan_gender['borrowers_count'].replace(0, 1)

    # We ensure that the ratio calculation avoids division by zero by replacing 0 with 1.

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

    # We're classifying loans into 3 categories: male-dominated, female-dominated, or mixed.

    #Done
    progress_bar.progress(100, None)

    return data

data = loading_dataset()

# Adding a description and motivation
st.markdown("""
#### Description and Motivation
This page contains an exploratory data analysis (EDA) of the KIVA loans dataset with visualizations to give a brief understanding of what the Dataset contains and what it is about. 

It has been chosen to include only the top 20 countries with the most loans, rather than including all countries, as it is beneficial for both exploratory data analysis (EDA) and building a supervised machine learning (SML) model to predict loans. 

By concentrating on the top 20 countries, the data becomes more reliable and representative of overall trends, ensuring that the insights drawn from the analysis and the predictions made by the model are more accurate. Including countries with very few loans introduces noise and sparse data, which may lead to less meaningful patterns and impact the model's performance. 

Additionally, narrowing the scope to the most relevant countries reduces the complexity of the model, making it easier to interpret and less prone to overfitting. This approach also ensures more efficient resource allocation, as the majority of loan activity occurs in these countries, making the insights and predictions more actionable for practical business use.

Overall, focusing on these key countries leads to a more effective EDA and an SML model that delivers clearer insights and better predictions.
""")


# PART 3: Setting up title and filter-sideheader
st.sidebar.header("Filters 📊")
#########################################################################################################################
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

# The first filter lets the user select a range of loan amounts using a slider.

# Filter the dataframe based on the selected loan amount range
filtered_data = data[
    (data['loan_amount'] >= selected_amount_range[0]) &
    (data['loan_amount'] <= selected_amount_range[1])
]

# We now filter the data to only include loans that fall within the selected amount range.

#########################################################################################################################
# GENDER SIDEBAR

# CREATE LIST OVER GENDERS
all_gender = data['gender_class'].unique().tolist()

# GENDER SIDEBAR MULTISELECT
selected_gender = st.sidebar.multiselect("Select Gender Group 🧑‍🧑‍🧒", all_gender, default=all_gender)

# This filter lets users select one or more gender groups to focus on.

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

# This allows users to filter the data by the sector of the loan.
#########################################################################################################################
# COUNTRY SIDEBAR

# CALCULATE TOP 20 COUNTRIES & CREATE A LIST
top_countries = data.groupby('country').size().nlargest(20).index.tolist()

# COUNTRY SIDEBAR MULTISELECT
selected_country = st.sidebar.multiselect(
    "Select Country 🇺🇳", top_countries, default=top_countries)

# Filtration of data based on sidebar
filtered_data = filtered_data[filtered_data['country'].isin(selected_country)]

# The country filter only shows the top 20 countries based on the number of loans.
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

# If the user doesn't select any option in one of the filters, a warning appears to notify them.

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
    chart = alt.Chart(filtered_data).mark

if visualization_option == "Records of Loans Issued By Sector & Country (Top 20 Countries)":
    # Bar chart for Records of Loans Issued By Sector (Top 20 Countries)
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[filtered_data['loan_amount'].min(), filtered_data['loan_amount'].max()])),  # Set x-axis to start at 0
        y='count()',
        color='sector',
    ).properties(
        title='Records of Loans Issued By Sector & Country (Top 20 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)


 # Bar chart for Records of Loans Issued By Country (Top 20 Countries)
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[filtered_data['loan_amount'].min(), filtered_data['loan_amount'].max()])),  # Force x-axis to start at 0
        y='count()',
        color='country',
    ).properties(
        title='Records of Loans Issued By Country Only (Top 20 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)


elif visualization_option == "KDE Plot - By Sector, Country & Total":
    # Starting with the KDE plot for loan amount by sector
    # KDE (Kernel Density Estimate) is used here to visualize the distribution of loan amounts across sectors
    plt.figure(figsize=(17, 7))  # Set the figure size
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='sector', fill=True, palette='gist_rainbow')  # KDE plot by sector, with color fill
    plt.xlabel('Loan Amount')  # Label x-axis
    plt.ylabel('Density')  # Label y-axis
    plt.title('KDE Plot of Loan Amount by Sector')  # Plot title
    plt.xlim(left=0)  # Ensure x-axis starts from 0 for a clearer view of the data
    st.pyplot(plt)  # Display the plot in Streamlit

    # Now for the KDE plot by country
    plt.figure(figsize=(17, 7))  # Create a new figure for the plot
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='country', fill=True, palette='gist_rainbow')  # KDE plot by country, same settings as above
    plt.xlabel('Loan Amount')  # x-axis label
    plt.ylabel('Density')  # y-axis label
    plt.title('KDE Plot of Loan Amount by Country')  # Plot title
    plt.xlim(left=0)  # Again, ensure x-axis starts at 0
    st.pyplot(plt)  # Display in Streamlit

    # Finally, a KDE plot of the total loan amount (no grouping by sector or country)
    plt.figure(figsize=(17, 7))  # Set up a new plot figure
    sns.kdeplot(data=filtered_data, x='loan_amount', fill=True, palette='gist_rainbow')  # Just show the overall density of loan amounts
    plt.xlabel('Loan Amount')  # Label x-axis
    plt.ylabel('Density')  # Label y-axis
    plt.title('KDE Plot of Total Loan Amount')  # Plot title
    plt.xlim(left=0)  # Make sure the x-axis starts from 0
    st.pyplot(plt)  # Display it in Streamlit

elif visualization_option == "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country":
    # Now let's create a stacked bar chart to show the mean loan amount by gender and sector
    # First we group the data by sector and gender, then calculate the mean loan amount for each group
    gender_sector = filtered_data.groupby(['sector', 'gender_class'])['loan_amount'].mean().unstack()
    gender_sector.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))  # Stacked bar chart, horizontal orientation
    plt.title('GENDER & SECTOR ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Sector')  # Title for the plot
    plt.ylabel('Sector')  # Label y-axis
    plt.xlabel('Mean Loan Amount')  # Label x-axis
    plt.xticks(rotation=0)  # No rotation for x-axis labels
    st.pyplot(plt)  # Display in Streamlit

    # Now we do the same thing, but this time for country and gender
    gender_country = filtered_data.groupby(['country', 'gender_class'])['loan_amount'].mean().unstack()
    gender_country.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))  # Similar stacked bar chart for country and gender
    plt.title('GENDER & COUNTRY ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Country')  # Title for this plot
    plt.ylabel('Country')  # Label y-axis
    plt.xlabel('Mean Loan Amount')  # Label x-axis
    plt.xticks(rotation=0)  # No label rotation again
    st.pyplot(plt)  # Display in Streamlit

elif visualization_option == "Heatmap of Average Loan by Sector & Country":
    # We want to show the average loan amount for each sector and country
    # First we pivot the data so that we have sectors as rows and countries as columns
    heatmap_data = filtered_data.pivot_table(index='sector', columns='country', values='loan_amount', aggfunc='mean')
    plt.figure(figsize=(20, 12))  # Set figure size for the heatmap
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")  # Heatmap with a coolwarm color map, showing the values as annotations
    plt.title('Heatmap of Average Loan by Sector & Country')  # Plot title
    st.pyplot(plt)  # Display in Streamlit

elif visualization_option == "Frequency of Funded Loans Over Time":
    # Here we are looking at how the number of funded loans changes over time
    time_data = filtered_data  # Using the filtered data we already have
    
    # CONVERTING 'funded_time' column to datetime format
    # This ensures that we can work with date and time values properly
    time_data['funded_time'] = pd.to_datetime(time_data['funded_time'])
    time_data.set_index('funded_time', inplace=True)  # Setting 'funded_time' as the index for easier time-based operations

    # Resample the data to a monthly frequency (can also be yearly, daily, etc.)
    # This gives us a count of funded loans per month
    funded_trend = time_data.resample('M').size()  # 'size()' counts the number of entries for each month

    # Create the plot
    plt.figure(figsize=(20, 6))  # Setting a large figure size for better visibility
    plt.plot(funded_trend, label='Total Funded Loans', color='blue')  # Plotting the trend with a label and color

    # Add labels, title, and legend
    plt.title('Frequency of Funded Loans Over Time')  # Title for the plot
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Number of Funded Loans')  # Label for the y-axis
    plt.legend()  # Display legend to help identify the line in the plot
    plt.grid(True)  # Adding a grid for easier readability
    
    # Display the plot in Streamlit
    st.pyplot(plt)  # Render the plot in the Streamlit app


# PART BONUS - DEBUGGING
# This section allows us to inspect the current state of the filtered data and selected options
with st.expander("DEBUGGING 🤓"):
    st.write("Selected sectors:", selected_sector)  # Displaying the sectors selected for filtering
    st.write("Selected countries:", selected_country)  # Displaying the countries selected for filtering
    st.write("Filtered data:", filtered_data)  # Showing the data that has been filtered based on user selections
