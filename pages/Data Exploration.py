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

    # Here we define the URLs where our dataset parts are hosted. 
    # They're hosted on GitHub, and we're going to download them in 3 parts.

    # Loading the urls into requests to download data
    progress_bar.progress(40, text="Downloading datasets...1/3")
    response1 = requests.get(url1)
    progress_bar.progress(55, text="Downloading datasets...2/3")
    response2 = requests.get(url2)
    progress_bar.progress(75, text="Downloading datasets...3/3")
    response3 = requests.get(url3)

    # With each response, we make progress and fetch the raw CSV files.

    # Loading partial datasets
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
    chart = alt.Chart(filtered_data).mark# PART 1: Function to load the data parts and use pd.concat to combine the 3 parts to one dataset
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

    # Here we define the URLs where our dataset parts are hosted. 
    # They're hosted on GitHub, and we're going to download them in 3 parts.

    # Loading the urls into requests to download data
    progress_bar.progress(40, text="Downloading datasets...1/3")
    response1 = requests.get(url1)
    progress_bar.progress(55, text="Downloading datasets...2/3")
    response2 = requests.get(url2)
    progress_bar.progress(75, text="Downloading datasets...3/3")
    response3 = requests.get(url3)

    # With each response, we make progress and fetch the raw CSV files.

    # Loading partial datasets
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
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[filtered_data['loan_amount'].min(), filtered_data['loan_amount'].max()])),  # Set x-axis to start at 0
        y='count()',
        color='sector',
    ).properties(
        title='Records of Loans Issued By Sector & Country (Top 20 Countries)'
    )
    st.altair_chart(chart, use_container_width=True)
# Here's where we create a bar chart showing the number of loans issued by country.
# We're using the 'loan_amount' for the x-axis, and we're making sure the x-axis starts from the lowest loan value to the highest.
    chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=[filtered_data['loan_amount'].min(), filtered_data['loan_amount'].max()])),  # Make the x-axis scale based on the data
        y='count()',  # y-axis will count how many loans each country issued
        color='country',  # Color the bars based on the country
        ).properties(
    title='Records of Loans Issued By Country Only (Top 20 Countries)'  # Title of the chart
)
    st.altair_chart(chart, use_container_width=True)  # Show the chart in Streamlit, using full width so it looks good

# If the user picked the option to display KDE plots
elif visualization_option == "KDE Plot - By Sector, Country & Total":

    # First up is the KDE plot for the loan amounts based on sector.
    # A KDE plot is kind of like a smoothed-out histogram—just a way to see the distribution of values.
    plt.figure(figsize=(17, 7))  # Setting the size of the chart
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='sector', fill=True, palette='gist_rainbow')  # We're plotting loan amounts, coloring by sector
    plt.xlabel('Loan Amount')  # X-axis label
    plt.ylabel('Density')  # Y-axis label, basically how concentrated the loans are
    plt.title('KDE Plot of Loan Amount by Sector')  # Title of the plot
    plt.xlim(left=0)  # Making sure the x-axis starts at 0
    st.pyplot(plt)  # Displaying the plot in Streamlit

    # Next is the KDE plot but this time, based on countries instead of sectors.
    plt.figure(figsize=(17, 7))  # Again, set the figure size
    sns.kdeplot(data=filtered_data, x='loan_amount', hue='country', fill=True, palette='gist_rainbow')  # Plotting by country
    plt.xlabel('Loan Amount')  # Label for the x-axis
    plt.ylabel('Density')  # Label for the y-axis
    plt.title('KDE Plot of Loan Amount by Country')  # Title of the chart
    plt.xlim(left=0)  # Keeping the x-axis starting at 0
    st.pyplot(plt)  # Show the plot in Streamlit

    # Lastly, a general KDE plot for the entire dataset (without splitting by sector or country)
    plt.figure(figsize=(17, 7))  # Again setting the size
    sns.kdeplot(data=filtered_data, x='loan_amount', fill=True, palette='gist_rainbow')  # Just plotting loan amounts for everything
    plt.xlabel('Loan Amount')  # X-axis label
    plt.ylabel('Density')  # Y-axis label
    plt.title('KDE Plot of Total Loan Amount')  # Title of the chart
    plt.xlim(left=0)  # Keep x-axis at 0
    st.pyplot(plt)  # Display in Streamlit

# This next part is for the stacked bar charts showing the mean loan amount by gender, sector, and country.
elif visualization_option == "Stacked Bar Chart - Mean Loan Amount by Gender, Sector & Country":

    # First, we're plotting by sector and gender. We group the data by sector and gender and take the mean of the loan amounts.
    gender_sector = filtered_data.groupby(['sector', 'gender_class'])['loan_amount'].mean().unstack()  # Group by sector and gender, then get mean loan
    gender_sector.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))  # Make a horizontal stacked bar chart
    plt.title('GENDER & SECTOR ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Sector')  # Chart title
    plt.ylabel('Sector')  # Y-axis label
    plt.xlabel('Mean Loan Amount')  # X-axis label
    plt.xticks(rotation=0)  # No rotation on x-axis labels
    st.pyplot(plt)  # Show the chart in Streamlit

    # Same thing, but now we group by country and gender.
    gender_country = filtered_data.groupby(['country', 'gender_class'])['loan_amount'].mean().unstack()  # Group by country and gender, get mean loan
    gender_country.plot(kind='barh', stacked=True, colormap='coolwarm', figsize=(14, 8))  # Horizontal stacked bar chart
    plt.title('GENDER & COUNTRY ONLY: Stacked Bar Chart of Mean Loan Amount by Gender & Country')  # Title of the chart
    plt.ylabel('Country')  # Y-axis label
    plt.xlabel('Mean Loan Amount')  # X-axis label
    plt.xticks(rotation=0)  # No rotation on x-axis labels
    st.pyplot(plt)  # Show the chart in Streamlit

# Now for a heatmap that shows the average loan amounts by sector and country.
elif visualization_option == "Heatmap of Average Loan by Sector & Country":

    # Pivot the data to make it easier to create a heatmap.
    heatmap_data = filtered_data.pivot_table(index='sector', columns='country', values='loan_amount', aggfunc='mean')  # Pivot table to get mean loan by sector and country
    plt.figure(figsize=(20, 12))  # Set figure size for the heatmap
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")  # Create a heatmap, with annotations (values on the chart) and a nice color scale
    plt.title('Heatmap of Average Loan by Sector & Country')  # Title of the heatmap
    st.pyplot(plt)  # Show the heatmap in Streamlit

# Finally, we have the time series plot that shows how the number of funded loans has changed over time.
elif visualization_option == "Frequency of Funded Loans Over Time":

    time_data = filtered_data  # Copy the filtered data
    
    # We need to make sure that the 'funded_time' column is in the right format (datetime) so we can work with it.
    time_data['funded_time'] = pd.to_datetime(time_data['funded_time'])  # Convert the 'funded_time' column to datetime
    time_data.set_index('funded_time', inplace=True)  # Set the index to 'funded_time' for easier resampling

    # We're resampling the data by month to smooth out the trend of funded loans.
    funded_trend = time_data.resample('M').size()  # Resample the data by month and count the number of loans

    # Plotting the time series.
    plt.figure(figsize=(20, 6))  # Set figure size for the plot
    plt.plot(funded_trend, label='Total Funded Loans', color='blue')  # Plot the number of funded loans over time

    # Add some labels, a title, and a legend so the plot makes sense.
    plt.title('Frequency of Funded Loans Over Time')  # Title of the plot
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Number of Funded Loans')  # Y-axis label
    plt.legend()  # Show the legend
    plt.grid(True)  # Add grid lines for better readability
    
    # Show the plot in Streamlit.
    st.pyplot(plt)

# And finally, there's a little debugging section here.
# It shows the filtered data, selected sectors, and selected countries, just so we can make sure everything's working as expected.
with st.expander("DEBUGGING 🤓"):
    st.write("Selected sectors:", selected_sector)  # Show the selected sectors
    st.write("Selected countries:", selected_country)  # Show the selected countries
    st.write("Filtered data:", filtered_data)  # Show the actual filtered data