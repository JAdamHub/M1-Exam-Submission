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

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap
from duckduckgo_search import DDGS
from sklearn.ensemble import VotingRegressor

### SML MODEL LIBS
from sklearn.preprocessing import LabelEncoder
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

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    ensemble_model = joblib.load('ensemble_model.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    
    return ensemble_model, scaler, ohe

ensemble_model, scaler, ohe = load_model_objects()

col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox('Sector', options=ohe.categories_[1])
    activity = st.selectbox('Activity', options=ohe.categories_[0])
    country = st.selectbox('Country', options=ohe.categories_[2])
    region = st.selectbox('Region', options=ohe.categories_[3])
    gender = st.radio('Gender', options=ohe.categories_[4])

with col2:
    borrowers_count = st.number_input('Number of Borrowers', min_value=0, max_value=30, value=1)
    funding_duration = st.number_input('Funding Duration Period (how long will you wait for funding?)', min_value=0, max_value=30, value=1)
    term_in_months = st.number_input('Terms in months (what amount of months will you pay back the loan)', min_value=1, max_value=145, value=12)

# Prediction button
if st.button('Predict Price 🚀'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'activity': [activity], 'sector': [sector], 'country': [country], 'region': [region], 'gender': [gender]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).todense(), 
                               columns=ohe.get_feature_names_out(['activity', 'sector', 'country', 'region', 'gender']))
    
    # Prepare numerical features
    num_features = pd.DataFrame({
        'borrowers_count': [borrowers_count],
        'funding_duration_days': [funding_duration],
        'term_in_months': [term_in_months],
    })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_price = ensemble_model.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted price per night", value=f'{round(predicted_price)} DKK')
    