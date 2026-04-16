import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

if len(sys.argv) > 1:
    raw_data_file = sys.argv[1]
else:
    print("[ERROR] No input file provided! Usage: python batch_preprocessing.py <filename>")
    sys.exit(1)

print(f"Loading data from: {raw_data_file}")
df_raw = pd.read_csv(raw_data_file)

#Feature engineering function
def engineer_features(data):
    df_engineered = data.copy()
    if 'application_date' in df_engineered.columns:
        df_engineered['application_date'] = pd.to_datetime(df_engineered['application_date'])
        df_engineered['app_year'] = df_engineered['application_date'].dt.year
        df_engineered['app_month'] = df_engineered['application_date'].dt.month
        df_engineered['app_day'] = df_engineered['application_date'].dt.day
        df_engineered = df_engineered.drop('application_date', axis=1)

    if 'loan_amount_requested' in df_engineered.columns and 'monthly_income' in df_engineered.columns:
        df_engineered['income_to_aid_ratio'] = df_engineered['loan_amount_requested'] / (df_engineered['monthly_income'] + 1e-5)

    return df_engineered

print("--- Starting Batch Preprocessing ---")

#Load the raw data for the current month
raw_data_file = 'api_data_2026_08.csv'
df_raw = pd.read_csv(raw_data_file)

#Apply Feature Engineering
df_engineered = engineer_features(df_raw)

columns_to_drop = ['application_id', 'customer_id', 'residential_address', 'fraud_type', 'loan_status']
df_engineered = df_engineered.drop(columns_to_drop, axis=1, errors='ignore')

#Load the Preprocessor Artifact
preprocessor = joblib.load('preprocessor.joblib')

#Transform the Data
X_processed_array = preprocessor.transform(df_engineered)

#Convert back to DataFrame and Save
feature_names = preprocessor.get_feature_names_out()
df_processed = pd.DataFrame(X_processed_array, columns=feature_names)

df_processed.to_csv('new_api_data_processed.csv', index=False)

#Simulation of audit team to create fraud flag
print("--- Simulating manual fraud auditing for new data ---")
df_processed['fraud_flag'] = np.random.choice([0, 1], size=len(df_processed), p=[0.98, 0.02])

df_processed.to_csv('feature_store_train_LATEST.csv', index=False)

# We also need a fake test set for evaluation. We will just use a split of this data.
train_latest, test_latest = train_test_split(df_processed, test_size=0.2, random_state=42)
train_latest.to_csv('feature_store_train_LATEST.csv', index=False)
test_latest.to_csv('feature_store_test_LATEST.csv', index=False)

print(f"--- Success! Processed {len(df_processed)} records into 'new_api_data_processed.csv' ---")