# M1 Exam Submission - KIVA MICROLOANS DATA EXPLORATION & LOAN PREDICTION 🪙💰

This project focuses on predicting loan amounts for Kiva microlending platform using machine learning techniques. It includes data exploration, model development, and a Streamlit web application for interactive predictions.

## Project Structure

- `data/`: Contains the dataset files & test data
  - `kiva_loans_part_0.csv` Partial dataset 1
  - `kiva_loans_part_1.csv` Partial dataset 2
  - `kiva_loans_part_2.csv` Partial dataset 3
  - `y_test.csv` Features variable testset
  - `X_test.csv` Target variable testset 
- `pages/`: Streamlit pages for the web application
  - `Data Exploration.py`
  - `Estimate Loan Amount.py`
- `M1-Exam_Data_Exploration.ipynb`: Jupyter notebook for data exploration
- `M1-Exam_Prediction_Model.ipynb`: Jupyter notebook for model development
- `app.py`: Main Streamlit application file
- `ensemble_model.joblib`: Saved machine learning ensemble model (30% Random Forest & 70% XGBoost)
- `requirements.txt`: List of required Python packages

## Setup and Installation

1. Clone this repository to your local machine.
2. Create a virtual environment (recommended):
   * Make sure to install python libraries from `requirements.txt`

